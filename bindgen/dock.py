import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bindgen.encoder import *
from bindgen.utils import *
from bindgen.nnutils import *
from bindgen.data import make_batch


class RefineDocker(ABModel):

    def __init__(self, args):
        super(RefineDocker, self).__init__(args)
        self.rstep = args.rstep
        self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.target_mpn = EGNNEncoder(args, update_X=False)
        self.hierarchical = args.hierarchical
        if args.hierarchical:
            self.struct_mpn = HierEGNNEncoder(args)
        else:
            self.struct_mpn = EGNNEncoder(args)

        self.W_x0 = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
        )
        self.U_x0 = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
        )
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
        cdist, _ = full_square_dist(bind_X, tgt_X, torch.ones_like(bind_X)[..., 0], torch.ones_like(tgt_X)[..., 0])
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

        tgt_mean = (tgt_X[:,:,1] * tgt_mask[...,None]).sum(dim=1) / tgt_mask[...,None].sum(dim=1).clamp(min=1e-4)
        bind_X = tgt_mean[:,None,None,:] + torch.rand_like(true_X)
        init_loss = 0

        # Refine
        dloss = vloss = rloss = iloss = closs = 0
        for t in range(self.rstep):
            # Interpolated label 
            ratio = (t + 1) / self.rstep
            label_X = true_X * ratio + bind_X.detach() * (1 - ratio)
            true_V = self.features._dihedrals(label_X)
            true_R, rmask_2D = inner_square_dist(label_X, true_A.clamp(max=1).float())
            true_D, mask_2D = self_square_dist(label_X, true_mask)
            true_C, cmask_2D = full_square_dist(label_X, tgt_X, true_A, tgt_A)
            inter_D, imask_2D = cross_square_dist(label_X, tgt_X, true_mask, tgt_mask)

            bind_V = self.features._dihedrals(bind_X)
            V = torch.cat([bind_V, tgt_V], dim=1).detach()
            X = torch.cat([bind_X, tgt_X], dim=1).detach()
            A = torch.cat([true_A, tgt_A], dim=1).detach()
            S = torch.cat([self.embedding(true_S), tgt_S], dim=1).detach()

            h_S = self.W_i(S)
            h, X = self.struct_mpn(X, V, h_S, A)
            bind_X = X[:, :N]

            dloss_t, vloss_t, rloss_t, iloss_t, closs_t = self.struct_loss(
                    bind_X, tgt_X, true_V, true_R, true_D, inter_D, true_C
            )
            vloss = vloss + vloss_t * true_mask
            dloss = dloss + dloss_t * mask_2D
            iloss = iloss + iloss_t * imask_2D
            rloss = rloss + rloss_t * rmask_2D
            closs = closs + closs_t * cmask_2D

        dloss = torch.sum(dloss) / mask_2D.sum()
        iloss = torch.sum(iloss) / imask_2D.sum()
        vloss = torch.sum(vloss) / true_mask.sum()
        if self.hierarchical:
            rloss = torch.sum(rloss) / rmask_2D.sum()
        else:
            rloss = torch.sum(rloss[:,:,:4,:4]) / rmask_2D[:,:,:4,:4].sum()

        loss = init_loss + dloss + iloss + vloss + rloss
        return ReturnType(loss=loss, bind_X=bind_X.detach(), handle=(tgt_X, tgt_A))
