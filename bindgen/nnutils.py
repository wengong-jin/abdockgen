import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bindgen.data import ALPHABET
from bindgen.protein_features import ProteinFeatures
from bindgen.utils import *


class Normalize(nn.Module):

    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)

        return gain * (x - mu) / (sigma + self.epsilon) + bias


class MPNNLayer(nn.Module):

    def __init__(self, num_hidden, num_in, dropout):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.Identity() #Normalize(num_hidden)
        self.W = nn.Sequential(
                nn.Linear(num_hidden + num_in, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, h_V, h_E, mask_attend):
        # h_V: [B, N, H]; h_E: [B, N, K, H]
        # mask_attend: [B, N, K]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)  # [B, N, K, H]
        h_message = self.W(h_EV) * mask_attend.unsqueeze(-1)
        dh = torch.mean(h_message, dim=-2)
        h_V = self.norm(h_V + self.dropout(dh))
        return h_V


class PosEmbedding(nn.Module):

    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    # E_idx: [B, N]
    def forward(self, E_idx):
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).cuda()
        angles = E_idx.unsqueeze(-1) * frequency.view((1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class AAEmbedding(nn.Module):

    def __init__(self):
        super(AAEmbedding, self).__init__()
        self.hydropathy = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
        self.volume = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
        self.charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
        self.polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
        self.embedding = torch.tensor([
            [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa],
            self.polarity[aa], self.acceptor[aa], self.donor[aa]]
            for aa in ALPHABET
        ]).cuda()

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = D_mu.view(1,1,-1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, :, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, :, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, :, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, :, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        B, N = x.size(0), x.size(1)
        aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs

    def soft_forward(self, x):
        B, N = x.size(0), x.size(1)
        aa_vecs = torch.matmul(x.reshape(B * N, -1), self.embedding).view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs)
        return rbf_vecs


class ABModel(nn.Module):

    def __init__(self, args):
        super(ABModel, self).__init__()
        self.k_neighbors = args.k_neighbors
        self.hidden_size = args.hidden_size
        self.embedding = AAEmbedding()
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.W_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

    def select_target(self, tgt_X, tgt_h, tgt_A, tgt_pos):
        max_len = max([len(pos) for pos in tgt_pos])
        xlist = [tgt_X[i, pos] for i,pos in enumerate(tgt_pos)]
        hlist = [tgt_h[i, pos] for i,pos in enumerate(tgt_pos)]
        alist = [tgt_A[i, pos] for i,pos in enumerate(tgt_pos)]
        tgt_X = [F.pad(x, (0,0,0,0,0,max_len-len(x))) for x in xlist]
        tgt_h = [F.pad(h, (0,0,0,max_len-len(h))) for h in hlist]
        tgt_A = [F.pad(a, (0,0,0,max_len-len(a))) for a in alist]
        return torch.stack(tgt_X, dim=0), torch.stack(tgt_h, dim=0), torch.stack(tgt_A, dim=0)
