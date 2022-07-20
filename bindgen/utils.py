import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

ReturnType = namedtuple('ReturnType',('loss','nll','ppl','bind_X','handle'), defaults=(None, None, None, None, None))


def kabsch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
    A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
    return A_aligned, R, t

# X: [B, N, 4, 3], R: [B, 3, 3], t: [B, 3]
def rigid_transform(X, R, t):
    B, N, L = X.size(0), X.size(1), X.size(2)
    X = X.reshape(B, N * L, 3)
    X = torch.bmm(R, X.transpose(1,2)).transpose(1,2) + t
    return X.view(B, N, L, 3)

# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd(A, B, mask):
    A_aligned, _, _ = kabsch(A, B)
    rmsd = ((A_aligned - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()

# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd_no_align(A, B, mask):
    rmsd = ((A - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()

def eig_coord(X, mask):
    D, mask_2D = self_square_dist(X, torch.ones_like(mask))
    return eig_coord_from_dist(D)

def eig_coord_from_dist(D):
    M = (D[:, :1, :] + D[:, :, :1] - D) / 2
    L, V = torch.linalg.eigh(M)
    L = torch.diag_embed(L)
    X = torch.matmul(V, L.clamp(min=0).sqrt())
    return X[:, :, -3:].detach()

def inner_square_dist(X, mask):
    L = mask.size(2)
    dX = X.unsqueeze(2) - X.unsqueeze(3)  # [B,N,1,L,3] - [B,N,L,1,3]
    mask_2D = mask.unsqueeze(2) * mask.unsqueeze(3)
    mask_2D = mask_2D * (1 - torch.eye(L)[None,None,:,:]).to(mask_2D)
    D = torch.sum(dX**2, dim=-1)
    return D * mask_2D, mask_2D

def self_square_dist(X, mask):
    X = X[:, :, 1] 
    dX = X.unsqueeze(1) - X.unsqueeze(2)  # [B, 1, N, 3] - [B, N, 1, 3]
    D = torch.sum(dX**2, dim=-1)
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, 1, N] x [B, N, 1]
    mask_2D = mask_2D * (1 - torch.eye(mask.size(1))[None,:,:]).to(mask_2D)
    return D, mask_2D

def cross_square_dist(X, Y, xmask, ymask):
    X, Y = X[:, :, 1], Y[:, :, 1]
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, N, 1, 3] - [B, 1, M, 3]
    D = torch.sum(dxy ** 2, dim=-1)
    mask_2D = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, N, 1] x [B, 1, M]
    return D, mask_2D

def full_square_dist(X, Y, XA, YA, contact=False, remove_diag=False):
    B, N, M, L = X.size(0), X.size(1), Y.size(1), Y.size(2)
    X = X.view(B, N * L, 3)
    Y = Y.view(B, M * L, 3)
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, NL, 1, 3] - [B, 1, ML, 3]
    D = torch.sum(dxy ** 2, dim=-1)
    D = D.view(B, N, L, M, L)
    D = D.transpose(2, 3).reshape(B, N, M, L*L)

    xmask = XA.clamp(max=1).float().view(B, N * L)
    ymask = YA.clamp(max=1).float().view(B, M * L)
    mask = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, NL, 1] x [B, 1, ML]
    mask = mask.view(B, N, L, M, L)
    mask = mask.transpose(2, 3).reshape(B, N, M, L*L)
    if remove_diag:
        mask = mask * (1 - torch.eye(N)[None,:,:,None]).to(mask)

    if contact:
        D = D + 1e6 * (1 - mask)
        return D.amin(dim=-1), mask.amax(dim=-1)
    else:
        return D, mask

""" Quaternion functions """

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (1e-4 + (quaternions * quaternions).sum(-1))
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(rot):
    if(rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot 

    k = [
        [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
        [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
        [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
        [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
    ]

    k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)
    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]

""" Graph functions """

def autoregressive_mask(E_idx):
    N_nodes = E_idx.size(1)
    ii = torch.arange(N_nodes).cuda()
    ii = ii.view((1, -1, 1))
    mask = E_idx - ii < 0
    return mask.float()

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn
