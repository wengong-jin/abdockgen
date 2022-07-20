import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os

from bindgen import *
from tqdm import tqdm

torch.set_num_threads(8)

def evaluate(model, loader, args):
    model.eval()
    total_nll = total = 0
    complex_rmsd, ab_bb_rmsd, ab_full_rmsd = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = make_batch(batch)
            out = model(*batch)

            bind_X, _, bind_A, _ = batch[0]
            tgt_X, tgt_A = out.handle
            bind_mask = bind_A.clamp(max=1).float()
            tgt_mask = tgt_A.clamp(max=1).float()
            idx1, idx2, idx3 = torch.nonzero(bind_mask, as_tuple=True)

            total_nll += out.nll.item() * bind_mask[..., 1].sum().item()
            total += bind_mask[..., 1].sum().item()

            true_X = torch.cat([bind_X, tgt_X], dim=1)
            pred_X = torch.cat([out.bind_X, tgt_X], dim=1)
            true_mask = torch.cat([bind_mask, torch.zeros_like(tgt_mask)], dim=1)
            rmsd = compute_rmsd(
                    pred_X[:, :, 1], true_X[:, :, 1], true_mask[:, :, 1]
            )
            complex_rmsd.extend(rmsd.tolist())
            rmsd = compute_rmsd(
                    out.bind_X[:, :, 1], bind_X[:, :, 1], bind_mask[:, :, 1]
            )
            ab_bb_rmsd.extend(rmsd.tolist())
            rmsd = compute_rmsd(
                    out.bind_X[idx1,idx2,idx3,:].view(1,-1,3),
                    bind_X[idx1,idx2,idx3,:].view(1,-1,3), 
                    bind_mask[idx1,idx2,idx3].view(1,-1),
            )
            ab_full_rmsd.extend(rmsd.tolist())

    return math.exp(total_nll / total), [sum(x) / len(x) for x in [complex_rmsd, ab_bb_rmsd, ab_full_rmsd]]


def build_model(args):
    if args.att_refine:
        model = AttRefineDecoder(args)
    elif args.no_target:
        model = UncondRefineDecoder(args)
    elif args.sequence:
        model = SequenceDecoder(args)
    else:
        model = CondRefineDecoder(args)
    return model.cuda()


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='data/rabd/train_data.jsonl')
parser.add_argument('--val_path', default='data/rabd/val_data.jsonl')
parser.add_argument('--test_path', default='data/rabd/test_data.jsonl')
parser.add_argument('--save_dir', default='ckpts/tmp')
parser.add_argument('--load_model', default=None)

parser.add_argument('--cdr', default='3')
parser.add_argument('--no_target', action='store_true', default=False)
parser.add_argument('--att_refine', action='store_true', default=False)
parser.add_argument('--hierarchical', action='store_true', default=False)
parser.add_argument('--sequence', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=100)
parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--L_target', type=int, default=20)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--clash_step', type=int, default=10)
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--clip_norm', type=float, default=1.0)

args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

all_data = []
for path in [args.train_path, args.val_path, args.test_path]:
    data = AntibodyComplexDataset(
            path,
            cdr_type=args.cdr,
            L_target=args.L_target,
    )
    all_data.append(data)

loader_train = ComplexLoader(all_data[0], batch_tokens=args.batch_tokens)
loader_val = ComplexLoader(all_data[1], batch_tokens=0)
loader_test = ComplexLoader(all_data[2], batch_tokens=0)

model = build_model(args)
optimizer = torch.optim.Adam(model.parameters())

if args.load_model:
    model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
    model = build_model(model_args)
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

print(type(model))
print('Training:{}, Validation:{}, Test:{}'.format(
    len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
)

best_ppl, best_epoch = 100, -1
for e in range(args.epochs):
    model.train()
    meter = 0

    for i,batch in enumerate(tqdm(loader_train)):
        optimizer.zero_grad()
        out = model(*make_batch(batch))
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meter += out.loss.item()
        if (i + 1) % args.print_iter == 0:
            meter /= args.print_iter
            print(f'[{i + 1}] Train Loss = {meter:.3f}')
            meter = 0

    val_ppl, val_rmsd = evaluate(model, loader_val, args)
    ckpt = (model.state_dict(), optimizer.state_dict(), args)
    torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
    print(f'Epoch {e}, Val PPL = {val_ppl:.3f}, Complex RMSD = {val_rmsd[0]:.3f}, Antibody backbone RMSD = {val_rmsd[1]:.3f}, All atom RMSD = {val_rmsd[2]:.3f}')

    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_epoch = e
        torch.save(ckpt, os.path.join(args.save_dir, f"model.best"))

if best_epoch >= 0:
    best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
    model.load_state_dict(torch.load(best_ckpt)[0])

test_ppl, test_rmsd = evaluate(model, loader_test, args)
print(f'Test PPL = {test_ppl:.3f}, Complex RMSD = {test_rmsd[0]:.3f}, Antibody backbone RMSD = {test_rmsd[1]:.3f}, All atom RMSD = {test_rmsd[2]:.3f}')
