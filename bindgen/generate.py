import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from bindgen.encoder import *
from bindgen.data import ALPHABET, ATOM_TYPES, RES_ATOM14
from bindgen.utils import *
from bindgen.nnutils import *


class CondRefineDecoder(ABModel):

    def __init__(self, args):
        super(CondRefineDecoder, self).__init__(args)
        self.hierarchical = args.hierarchical
        self.residue_atom14 = torch.tensor([
                [ATOM_TYPES.index(a) for a in atoms] for atoms in RES_ATOM14
        ]).cuda()

        self.W_s0 = nn.Sequential(
                PosEmbedding(32),
                nn.Linear(32, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, len(ALPHABET))
        )
        self.W_x0 = nn.Sequential(
                PosEmbedding(32),
                nn.Linear(32, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
        )
        self.U_x0 = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
        )
        self.W_s = nn.Linear(args.hidden_size, len(ALPHABET))
        self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.target_mpn = EGNNEncoder(args, update_X=False)

        if args.hierarchical:
            self.struct_mpn = HierEGNNEncoder(args)
            self.seq_mpn = HierEGNNEncoder(args, update_X=False, backbone_CA_only=False)
        else:
            self.struct_mpn = EGNNEncoder(args)
            self.seq_mpn = EGNNEncoder(args, update_X=False)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def struct_loss(self, bind_X, tgt_X, true_V, true_R, true_D, inter_D, true_C):
        # dihedral loss
        bind_V = self.features._dihedrals(bind_X)
        vloss = self.mse_loss(bind_V, true_V).sum(dim=-1)
        # local loss
        rdist = bind_X.unsqueeze(2) - bind_X.unsqueeze(3)
        rdist = torch.sum(rdist ** 2, dim=-1)
        rloss = self.huber_loss(rdist, true_R) + 10 * F.relu(1.5 - rdist)
        # full loss
        cdist, _ = full_square_dist(bind_X, bind_X, torch.ones_like(bind_X)[..., 0], torch.ones_like(bind_X)[..., 0])
        closs = self.huber_loss(cdist, true_C) + 10 * F.relu(1.5 - cdist)
        # alpha carbon
        bind_X, tgt_X = bind_X[:, :, 1], tgt_X[:, :, 1]
        # CDR self distance
        dist = bind_X.unsqueeze(1) - bind_X.unsqueeze(2)
        dist = torch.sum(dist ** 2, dim=-1)
        dloss = self.huber_loss(dist, true_D) + 10 * F.relu(14.4 - dist)
        # inter distance
        idist = bind_X.unsqueeze(2) - tgt_X.unsqueeze(1)
        idist = torch.sum(idist ** 2, dim=-1)
        iloss = self.huber_loss(idist, inter_D) + 10 * F.relu(14.4 - idist)
        return dloss, vloss, rloss, iloss, closs

    def forward(self, binder, target, surface):
        true_X, true_S, true_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface

        # Encode target
        tgt_S = self.embedding(tgt_S)
        tgt_V = self.features._dihedrals(tgt_X)
        tgt_h, _ = self.target_mpn(tgt_X, tgt_V, self.U_i(tgt_S), tgt_A)
        _, tgt_S, _ = self.select_target(tgt_X, tgt_S, tgt_A, tgt_surface)
        tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface)
        tgt_V = self.features._dihedrals(tgt_X)

        B, N, M = true_S.size(0), true_S.size(1), tgt_X.size(1)
        true_mask = true_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        true_h_S = self.embedding(true_S)

        # Initial state
        pos_vecs = torch.arange(N)[None,:].expand(B, -1).cuda()
        logits = self.W_s0(pos_vecs)
        init_prob = F.softmax(logits, dim=-1)
        init_loss = self.ce_loss(logits.view(-1,len(ALPHABET)), true_S.view(-1))
        init_loss = torch.sum(init_loss * true_mask.view(-1))

        bind_S = self.embedding.soft_forward(init_prob)
        bind_A = true_A.clone()
        bind_A[:,:,4:] = 0  # only backbone atoms

        # Initial fold
        all_h = torch.cat([self.W_x0(pos_vecs), self.U_x0(tgt_h)], dim=1) * 10
        dX = all_h[:,:,None] - all_h[:,None,:]  # [B,N+M,N+M,H]
        dist = torch.sum(dX ** 2, dim=-1)

        all_X = torch.cat([true_X, tgt_X], dim=1)
        all_mask = torch.cat([true_mask, tgt_mask], dim=1)
        true_D, mask_2D = self_square_dist(all_X, all_mask)
        init_fold_loss = self.huber_loss(dist, true_D) * mask_2D
        init_loss = init_loss + init_fold_loss.sum() / mask_2D.sum()

        init_bb = [None] * 14
        with torch.no_grad():
            dist = dist.detach().clone()
            dist[:, N:, N:] = true_D[:, N:, N:]  # epitope dist is known
            init_X = eig_coord_from_dist(dist)
            tgt_all_mask = tgt_A.clamp(max=1).float().unsqueeze(-1)
            tgt_ca_fill = tgt_X[:, :, 1:2] * (1 - tgt_all_mask)
            tgt_X_fill = tgt_X * tgt_all_mask + tgt_ca_fill
            for i in range(14):
                _, R, t = kabsch(init_X[:, N:, :].clone(), tgt_X_fill[:, :, i])
                init_bb[i] = rigid_transform(init_X.unsqueeze(2), R, t)
                init_bb[i] = init_bb[i][:, :N, 0].clone()
            bind_X = torch.stack(init_bb, dim=2).clone()

        # Refine
        dloss = vloss = rloss = iloss = sloss = closs = 0
        for t in range(4):
            bind_V = self.features._dihedrals(bind_X.detach())
            V = torch.cat([bind_V, tgt_V], dim=1).detach()
            X = torch.cat([bind_X, tgt_X], dim=1).detach()
            A = torch.cat([bind_A, tgt_A], dim=1).detach()
            h_S = self.W_i(torch.cat([bind_S, tgt_S], dim=1))
            h, X = self.struct_mpn(X, V, h_S, A)
            bind_X = X[:, :N]

            # Interpolated label 
            ratio = (t + 1) / 4
            label_X = true_X * ratio + bind_X.detach() * (1 - ratio)
            true_V = self.features._dihedrals(label_X)
            true_R, rmask_2D = inner_square_dist(label_X, bind_A.clamp(max=1).float())
            true_D, mask_2D = self_square_dist(label_X, true_mask)
            true_C, cmask_2D = full_square_dist(label_X, label_X, bind_A, bind_A)
            inter_D, imask_2D = cross_square_dist(label_X, tgt_X, true_mask, tgt_mask)

            dloss_t, vloss_t, rloss_t, iloss_t, closs_t = self.struct_loss(
                    bind_X, tgt_X, true_V, true_R, true_D, inter_D, true_C
            )
            vloss = vloss + vloss_t * true_mask
            dloss = dloss + dloss_t * mask_2D
            iloss = iloss + iloss_t * imask_2D
            rloss = rloss + rloss_t * rmask_2D
            closs = closs + closs_t * cmask_2D

        for t in range(N):
            bind_V = self.features._dihedrals(bind_X.detach())
            V = torch.cat([bind_V, tgt_V], dim=1).detach()
            X = torch.cat([bind_X, tgt_X], dim=1).detach()
            A = torch.cat([bind_A, tgt_A], dim=1).detach()

            # Predict residue t
            h_S = self.W_i(torch.cat([bind_S, tgt_S], dim=1))
            h, _ = self.seq_mpn(X, V, h_S, A)

            logits = self.W_s(h[:, t])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * true_mask[:, t])

            # Teacher forcing
            bind_S = bind_S.clone()
            bind_S[:, t] = true_h_S[:, t]
            bind_S = bind_S.clone()

        sloss = sloss / true_mask.sum()
        dloss = torch.sum(dloss) / mask_2D.sum()
        iloss = torch.sum(iloss) / imask_2D.sum()
        vloss = torch.sum(vloss) / true_mask.sum()
        if self.hierarchical:
            rloss = torch.sum(rloss) / rmask_2D.sum()
            closs = torch.sum(closs) / cmask_2D.sum()
        else:
            rloss = torch.sum(rloss[:,:,:4,:4]) / rmask_2D[:,:,:4,:4].sum()
            closs = 0

        loss = init_loss + sloss + (dloss + iloss + vloss + rloss + closs) / N
        return ReturnType(loss=loss, nll=sloss, bind_X=bind_X.detach(), handle=(tgt_X, tgt_A))

    def generate(self, target, surface):
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface
        B, N = len(tgt_X), len(bind_surface[0])  # assume equal length

        # Encode target (assumes same target)
        tgt_X, tgt_S, tgt_A = tgt_X[:1], tgt_S[:1], tgt_A[:1]
        tgt_S = self.embedding(tgt_S)
        tgt_V = self.features._dihedrals(tgt_X)
        tgt_h, _ = self.target_mpn(tgt_X, tgt_V, self.U_i(tgt_S), tgt_A)
        _, tgt_S, _ = self.select_target(tgt_X, tgt_S, tgt_A, tgt_surface[:1])
        tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface[:1])
        tgt_V = self.features._dihedrals(tgt_X)

        # Copy across batch dimension
        tgt_X = tgt_X.expand(B,-1,-1,-1)
        tgt_h = tgt_h.expand(B,-1,-1)
        tgt_A = tgt_A.expand(B,-1,-1)
        tgt_S = tgt_S.expand(B,-1,-1)
        tgt_V = tgt_V.expand(B,-1,-1)

        true_mask = torch.zeros(B, N).cuda()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask = torch.cat([true_mask, tgt_mask], dim=1)

        # Encode CDR
        pos_vecs = torch.arange(N)[None,:].expand(B, -1).cuda()
        logits = self.W_s0(pos_vecs)
        init_prob = F.softmax(logits, dim=-1)

        # Initial state
        bind_S = self.embedding.soft_forward(init_prob)
        bind_X = torch.zeros(B, N, 14, 3).cuda()
        bind_I = torch.zeros(B, N).cuda().long()
        bind_A = torch.zeros(B, N, 14).cuda().long()
        for i in range(4):
            bind_A[:,:,i] = i + 1  # backbone atoms

        # Initial fold
        all_h = torch.cat([self.W_x0(pos_vecs), self.U_x0(tgt_h)], dim=1) * 10
        dX = all_h[:,:,None] - all_h[:,None,:]  # [B,N+M,N+M,H]
        dist = torch.sum(dX ** 2, dim=-1)

        init_bb = [None] * 14
        with torch.no_grad():
            dist = dist.detach().clone()
            tgt_D, _ = self_square_dist(tgt_X, tgt_mask)
            dist[:, N:, N:] = tgt_D
            init_X = eig_coord_from_dist(dist)
            tgt_all_mask = tgt_A.clamp(max=1).float().unsqueeze(-1)
            tgt_ca_fill = tgt_X[:, :, 1:2] * (1 - tgt_all_mask)
            tgt_X_fill = tgt_X * tgt_all_mask + tgt_ca_fill
            for i in range(14):
                _, R, t = kabsch(init_X[:, N:, :].clone(), tgt_X_fill[:, :, i])
                init_bb[i] = rigid_transform(init_X.unsqueeze(2), R, t)
                init_bb[i] = init_bb[i][:, :N, 0].clone()
            bind_X = torch.stack(init_bb, dim=2).clone()

        # Refine
        sloss = 0.
        for t in range(N):
            bind_V = self.features._dihedrals(bind_X)
            V = torch.cat([bind_V, tgt_V], dim=1).detach()
            X = torch.cat([bind_X, tgt_X], dim=1).detach()
            A = torch.cat([bind_A, tgt_A], dim=1).detach()

            # Predict residue t
            h_S = self.W_i(torch.cat([bind_S, tgt_S], dim=1))
            h, _ = self.seq_mpn(X, V, h_S, A)

            logits = self.W_s(h[:, t])
            prob = F.softmax(logits, dim=-1)
            bind_I[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            bind_S[:, t] = self.embedding(bind_I)[:, t]
            sloss = sloss + self.ce_loss(logits, bind_I[:, t])
            #bind_A[:, t] = self.residue_atom14[bind_I[:, t]]

            # Predict structure
            X = torch.cat([bind_X, tgt_X], dim=1).detach()
            A = torch.cat([bind_A, tgt_A], dim=1).detach()
            h_S = self.W_i(torch.cat([bind_S, tgt_S], dim=1))
            h, X = self.struct_mpn(X, V, h_S, A)
            bind_X = X[:, :N]

        S = bind_I.tolist()
        S = [''.join([ALPHABET[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return ReturnType(handle=S, ppl=ppl, bind_X=bind_X.detach())


class AttRefineDecoder(ABModel):

    def __init__(self, args):
        super(AttRefineDecoder, self).__init__(args)
        self.W_x = nn.Linear(args.hidden_size, 42)
        self.W_s = nn.Linear(args.hidden_size * 2, len(ALPHABET))
        self.struct_mpn = EGNNEncoder(args, update_X=False)
        self.seq_mpn = EGNNEncoder(args, update_X=False)
        self.W_x0 = nn.Sequential(
                PosEmbedding(32),
                nn.Linear(32, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 42)
        )
        self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.target_mpn = EGNNEncoder(args, update_X=False)
        self.W_att = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # Q: [B, H], context: [B, M, H]
    def attention(self, Q, context, cmask):
        att = torch.bmm(Q[:, None], context.transpose(1, 2))  # [B, N, M]
        att = att - 1e6 * (1 - cmask.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, context)  # [B, 1, M] * [B, M, H]
        return out.squeeze(1)

    def struct_loss(self, X, mask, true_D, true_V, true_R, true_C):
        D, _ = self_square_dist(X, mask[:,:,1])
        R, _ = inner_square_dist(X, mask)
        C, _ = full_square_dist(X, X, torch.ones_like(X)[..., 0], torch.ones_like(X)[..., 0])
        V = self.features._dihedrals(X)
        dloss = self.huber_loss(D, true_D)
        vloss = self.mse_loss(V, true_V).sum(dim=-1)
        rloss = self.huber_loss(R, true_R)
        closs = self.huber_loss(C, true_C)
        return dloss, vloss, rloss, closs

    def forward(self, binder, target, surface):
        true_X, true_S, true_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface

        # Encode target
        tgt_h_S = self.U_i(self.embedding(tgt_S))
        tgt_V = self.features._dihedrals(tgt_X)
        tgt_h, _ = self.target_mpn(tgt_X, tgt_V, tgt_h_S, tgt_A)
        tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface)
        tgt_V = self.features._dihedrals(tgt_X)

        B, N, M = true_S.size(0), true_S.size(1), tgt_X.size(1)
        true_mask = true_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()

        # Ground truth 
        B, N, L = true_A.size(0), true_A.size(1), true_A.size(2)
        true_mask = true_A.clamp(max=1).float()
        true_V = self.features._dihedrals(true_X)
        true_D, mask_2D = self_square_dist(true_X, true_mask[:,:,1])
        true_R, rmask_2D = inner_square_dist(true_X, true_mask)
        true_C, cmask_2D = full_square_dist(true_X, true_X, true_A, true_A)

        # Initial coords
        pos_vecs = torch.arange(N)[None,:].expand(B, -1).cuda()
        X = self.W_x0(pos_vecs).view(B, N, L, 3)
        dloss, vloss, rloss, closs = self.struct_loss(X, true_mask, true_D, true_V, true_R, true_C)

        # Initial residues
        S = torch.zeros(B, N).cuda().long()
        sloss = 0.

        for t in range(N):
            X = X.detach().clone()
            V = self.features._dihedrals(X)
            # Predict residue t
            h_S = self.W_i(self.embedding(S))
            h, _ = self.seq_mpn(X, V, h_S, true_A)
            h_att = self.attention(h[:, t], tgt_h, tgt_mask)
            h = torch.cat((h_att, h[:, t]), dim=-1)
            logits = self.W_s(h)
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * true_mask[:,t,1])

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Iterative refinement
            h_S = self.W_i(self.embedding(S))
            h, _ = self.struct_mpn(X, V, h_S, true_A)
            X = self.W_x(h).view(B, N, L, 3)
            X = X * true_mask[...,None]
            dloss_t, vloss_t, rloss_t, closs_t = self.struct_loss(X, true_mask, true_D, true_V, true_R, true_C)
            dloss += dloss_t
            vloss += vloss_t
            rloss += rloss_t
            closs += closs_t

        sloss = sloss / true_mask[:,:,1].sum()
        dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
        vloss = torch.sum(vloss * true_mask[:,:,1]) / true_mask[:,:,1].sum()
        rloss = torch.sum(rloss * rmask_2D) / rmask_2D.sum()
        closs = torch.sum(closs * cmask_2D) / cmask_2D.sum()
        loss = sloss + (dloss + vloss + rloss + closs) / N
        return ReturnType(loss=loss, nll=sloss, bind_X=X.detach(), handle=(tgt_X, tgt_A))

    def generate(self, target, surface):
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface
        B, N = len(tgt_X), len(bind_surface[0])  # assume equal length

        # Encode target (assumes same target)
        tgt_X, tgt_S, tgt_A = tgt_X[:1], tgt_S[:1], tgt_A[:1]
        tgt_h_S = self.U_i(self.embedding(tgt_S))
        tgt_V = self.features._dihedrals(tgt_X)
        tgt_h, _ = self.target_mpn(tgt_X, tgt_V, tgt_h_S, tgt_A)
        tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface[:1])

        # Copy across batch dimension
        tgt_X = tgt_X.expand(B,-1,-1,-1)
        tgt_h = tgt_h.expand(B,-1,-1)
        tgt_A = tgt_A.expand(B,-1,-1)

        true_A = torch.ones(B, N, 14).cuda()
        true_mask = torch.ones(B, N).cuda()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()

        # Initial coords
        pos_vecs = torch.arange(N)[None,:].expand(B, -1).cuda()
        X = self.W_x0(pos_vecs).view(B, N, 14, 3)

        # Initial residues
        S = torch.zeros(B, N).cuda().long()
        sloss = 0.

        for t in range(N):
            X = X.detach().clone()
            V = self.features._dihedrals(X)
            # Predict residue t
            h_S = self.W_i(self.embedding(S))
            h, _ = self.seq_mpn(X, V, h_S, true_A)
            h_att = self.attention(h[:, t], tgt_h, tgt_mask)
            h = torch.cat((h_att, h[:, t]), dim=-1)
            logits = self.W_s(h)
            prob = F.softmax(logits, dim=-1)
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            sloss = sloss + self.ce_loss(logits, S[:, t])

            # Iterative refinement
            h_S = self.W_i(self.embedding(S))
            h, _ = self.struct_mpn(X, V, h_S, true_A)
            X = self.W_x(h).view(B, N, 14, 3)

        S = S.tolist()
        S = [''.join([ALPHABET[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return ReturnType(handle=S, ppl=ppl, bind_X=X.detach())


class UncondRefineDecoder(ABModel):

    def __init__(self, args):
        super(UncondRefineDecoder, self).__init__(args)
        self.W_x = nn.Linear(args.hidden_size, 42)
        self.W_s = nn.Linear(args.hidden_size, len(ALPHABET))
        self.struct_mpn = EGNNEncoder(args, update_X=False)
        self.seq_mpn = EGNNEncoder(args, update_X=False)
        self.W_x0 = nn.Sequential(
                PosEmbedding(32),
                nn.Linear(32, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 42)
        )
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def struct_loss(self, X, mask, true_D, true_V, true_R, true_C):
        D, _ = self_square_dist(X, mask[:,:,1])
        R, _ = inner_square_dist(X, mask)
        C, _ = full_square_dist(X, X, torch.ones_like(X)[..., 0], torch.ones_like(X)[..., 0])
        V = self.features._dihedrals(X)
        dloss = self.huber_loss(D, true_D)
        vloss = self.mse_loss(V, true_V).sum(dim=-1)
        rloss = self.huber_loss(R, true_R)
        closs = self.huber_loss(C, true_C)
        return dloss, vloss, rloss, closs

    def forward(self, binder, target, surface):
        true_X, true_S, true_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface

        # Dummy operation
        tgt_h = self.embedding(tgt_S)
        tgt_X, _, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface)

        # Ground truth 
        B, N, L = true_A.size(0), true_A.size(1), true_A.size(2)
        true_mask = true_A.clamp(max=1).float()
        true_V = self.features._dihedrals(true_X)
        true_D, mask_2D = self_square_dist(true_X, true_mask[:,:,1])
        true_R, rmask_2D = inner_square_dist(true_X, true_mask)
        true_C, cmask_2D = full_square_dist(true_X, true_X, true_A, true_A)

        # Initial coords
        pos_vecs = torch.arange(N)[None,:].expand(B, -1).cuda()
        X = self.W_x0(pos_vecs).view(B, N, L, 3)
        dloss, vloss, rloss, closs = self.struct_loss(X, true_mask, true_D, true_V, true_R, true_C)

        # Initial residues
        S = torch.zeros(B, N).cuda().long()
        sloss = 0.

        for t in range(N):
            X = X.detach().clone()
            V = self.features._dihedrals(X)
            # Predict residue t
            h_S = self.W_i(self.embedding(S))
            h, _ = self.seq_mpn(X, V, h_S, true_A)
            logits = self.W_s(h[:, t])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * true_mask[:,t,1])

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Iterative refinement
            h_S = self.W_i(self.embedding(S))
            h, _ = self.struct_mpn(X, V, h_S, true_A)
            X = self.W_x(h).view(B, N, L, 3)
            X = X * true_mask[...,None]
            dloss_t, vloss_t, rloss_t, closs_t = self.struct_loss(X, true_mask, true_D, true_V, true_R, true_C)
            dloss += dloss_t
            vloss += vloss_t
            rloss += rloss_t
            closs += closs_t

        sloss = sloss / true_mask[:,:,1].sum()
        dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
        vloss = torch.sum(vloss * true_mask[:,:,1]) / true_mask[:,:,1].sum()
        rloss = torch.sum(rloss * rmask_2D) / rmask_2D.sum()
        closs = torch.sum(closs * cmask_2D) / cmask_2D.sum()
        loss = sloss + (dloss + vloss + rloss + closs) / N
        return ReturnType(loss=loss, nll=sloss, bind_X=X.detach(), handle=(tgt_X, tgt_A))

    def generate(self, target, surface):
        bind_surface, _ = surface
        B, N = len(bind_surface), len(bind_surface[0])  # assume equal length

        true_A = torch.ones(B, N, 14).cuda()
        true_mask = torch.ones(B, N).cuda()

        # Initial coords
        pos_vecs = torch.arange(N)[None,:].expand(B, -1).cuda()
        X = self.W_x0(pos_vecs).view(B, N, 14, 3)

        # Initial residues
        S = torch.zeros(B, N).cuda().long()
        sloss = 0.

        for t in range(N):
            X = X.detach().clone()
            V = self.features._dihedrals(X)
            # Predict residue t
            h_S = self.W_i(self.embedding(S))
            h, _ = self.seq_mpn(X, V, h_S, true_A)
            logits = self.W_s(h[:, t])
            prob = F.softmax(logits, dim=-1)
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            sloss = sloss + self.ce_loss(logits, S[:, t])

            # Iterative refinement
            h_S = self.W_i(self.embedding(S))
            h, _ = self.struct_mpn(X, V, h_S, true_A)
            X = self.W_x(h).view(B, N, 14, 3)

        S = S.tolist()
        S = [''.join([ALPHABET[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return ReturnType(handle=S, ppl=ppl, bind_X=X.detach())


class SequenceDecoder(ABModel):

    def __init__(self, args):
        super(SequenceDecoder, self).__init__(args)
        self.no_target = args.no_target
        self.W_s = nn.Linear(args.hidden_size, len(ALPHABET))
        self.seq_rnn = SRUpp(
                args.hidden_size,
                args.hidden_size,
                args.hidden_size // 2,
                num_layers=args.depth,
                dropout=args.dropout,
                bidirectional=False,
        )
        if not self.no_target:
            self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)
            self.W_s = nn.Linear(args.hidden_size * 2, len(ALPHABET))
            self.target_mpn = EGNNEncoder(args, update_X=False)
            self.W_att = nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size),
                    nn.ReLU(),
            )
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # Q: [B, H], context: [B, M, H]
    def attention(self, Q, context, cmask):
        att = torch.bmm(Q[:, None], context.transpose(1, 2))  # [B, N, M]
        att = att - 1e6 * (1 - cmask.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, context)  # [B, 1, M] * [B, M, H]
        return out.squeeze(1)

    def forward(self, binder, target, surface):
        true_X, true_S, true_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface
        B, N, M = true_S.size(0), true_S.size(1), tgt_S.size(1)

        # Encode target
        if not self.no_target:
            tgt_h_S = self.U_i(self.embedding(tgt_S))
            tgt_V = self.features._dihedrals(tgt_X)
            tgt_h, _ = self.target_mpn(tgt_X, tgt_V, tgt_h_S, tgt_A)
            tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface)
            tgt_mask = tgt_A[:,:,1].clamp(max=1).float()

        true_mask = true_A[:,:,1].clamp(max=1).float()
        S = torch.zeros_like(true_S)
        sloss = 0.

        for t in range(N):
            h_S = self.W_i(self.embedding(S))
            h, _, _ = self.seq_rnn(
                    h_S.transpose(0, 1), 
                    mask_pad=(~true_mask.transpose(0, 1).bool()),
            )
            h = h.transpose(0, 1)
            if self.no_target:
                logits = self.W_s(h[:, t])
            else:
                h_att = self.attention(h[:, t], tgt_h, tgt_mask)
                h = torch.cat((h_att, h[:, t]), dim=-1)
                logits = self.W_s(h)

            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * true_mask[:, t])
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

        loss = sloss.sum() / true_mask.sum()
        return ReturnType(loss=loss, nll=loss, bind_X=true_X, handle=(tgt_X, tgt_A))

    def generate(self, target, surface):
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface
        B, N = len(tgt_X), len(bind_surface[0])

        # Encode target (assumes same target)
        if not self.no_target:
            tgt_X, tgt_S, tgt_A = tgt_X[:1], tgt_S[:1], tgt_A[:1]
            tgt_h_S = self.U_i(self.embedding(tgt_S))
            tgt_V = self.features._dihedrals(tgt_X)
            tgt_h, _ = self.target_mpn(tgt_X, tgt_V, tgt_h_S, tgt_A)
            tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface[:1])

            # Copy across batch dimension
            tgt_X = tgt_X.expand(B,-1,-1,-1)
            tgt_h = tgt_h.expand(B,-1,-1)
            tgt_A = tgt_A.expand(B,-1,-1)
            tgt_mask = tgt_A[:,:,1].clamp(max=1).float()

        true_mask = torch.ones(B, N).cuda()
        S = torch.zeros(B, N).long().cuda()
        sloss = 0.

        for t in range(N):
            h_S = self.W_i(self.embedding(S))
            h, _, _ = self.seq_rnn(
                    h_S.transpose(0, 1), 
                    mask_pad=(~true_mask.transpose(0, 1).bool()),
            )
            h = h.transpose(0, 1)
            if self.no_target:
                logits = self.W_s(h[:, t])
            else:
                h_att = self.attention(h[:, t], tgt_h, tgt_mask)
                h = torch.cat((h_att, h[:, t]), dim=-1)
                logits = self.W_s(h)

            prob = F.softmax(logits, dim=-1)
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            sloss = sloss + self.ce_loss(logits, S[:, t])

        S = S.tolist()
        S = [''.join([ALPHABET[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return ReturnType(handle=S, ppl=ppl)
