import torch
import torch.nn as nn
import torch.optim as optim

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os

from bindgen import *
from tqdm import tqdm
from copy import deepcopy

import pdbfixer
import openmm
import multiprocessing
import biotite.structure as struc
from biotite.structure import AtomArray, Atom
from biotite.structure.io import save_structure
from biotite.structure.io.pdb import PDBFile
from pyfoldx.structure import Structure


ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms
torch.set_num_threads(8)


def print_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aid = ALPHABET.index(aaname)
        aaname = RESTYPE_1to3[aaname]
        for j,atom in enumerate(RES_ATOM14[aid]):
            if atom != '':
                atom = Atom(coord[i, j], chain_id=chain, res_id=idx, atom_name=atom, res_name=aaname, element=atom[0])
                array.append(atom)
    return array


def openmm_relax(pdb_file, tolerance=2.39, use_gpu=False):
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    tolerance = tolerance * ENERGY
    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getKineticEnergy() + state.getPotentialEnergy()

    with open(pdb_file, "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True
        )
    return energy


def compute_energy(path):
    obj = Structure("pred", path=path)
    obj = obj.repair(other_parameters={"repair_Interface": "ONLY"}, verbose=False)
    energy = obj.getInterfaceEnergy(verbose=False)['Interaction Energy'].values[0]
    return float(energy)


if __name__ == "__main__":
    # Example usage:
    # python cal_energy.py ckpts/HERN_dock.ckpt data/rabd/test_data.jsonl generated.txt
    model_ckpt, _, args = torch.load(sys.argv[1])
    model = RefineDocker(args).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()

    data = AntibodyComplexDataset(
            sys.argv[2],
            cdr_type=args.cdr,
            L_target=args.L_target,
    )

    # generated.txt contains designed sequences output from generate.py
    # format: pdb original_CDR3 generated_CDR3
    # each line: 4ffv TRFRDVFFDV ARDYYGYFDV ...
    decoys = {}
    with open(sys.argv[3]) as f:
        for line in f:
            pdb, ref, seq = line.split()[:3]
            if pdb not in decoys:
                decoys[pdb] = set()
            decoys[pdb].add((ref, seq))

    with torch.no_grad():
        for ab in data.data:
            pdb = ab['pdb']
            path = os.path.join('outputs', f'{pdb}_true.pdb')
            X = np.array(ab['binder_coords'])
            Y = np.array(ab['antigen_coords'])
            path = os.path.join('outputs', f'{pdb}_pred.pdb')
            array1 = print_pdb(X, ab['binder_seq'], 'H')
            array2 = print_pdb(Y, ab['antigen_seq'], 'A')
            array = struc.array(array2 + array1)
            save_structure(path, array)
            openmm_relax(path, use_gpu=True)
            energy = compute_energy(path)
            print(pdb, 'true', ab['binder_seq'], energy)

    with torch.no_grad():
        for ab in data.data:
            pdb = ab['pdb']
            cands = []
            acc = []
            for i,(ref,seq) in enumerate(tqdm(decoys[pdb])):
                ab = deepcopy(ab)
                ab['binder_seq'] = seq
                ab['binder_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in seq]
                )
                batch = make_batch([ab])
                batch[0][0].fill_(0)  # remove ground truth
                out = model(*batch)

                X = out.bind_X[0].cpu().numpy()
                Y = np.array(ab['antigen_coords'])
                path = os.path.join('outputs', f'{pdb}_{i}.pdb')
                array1 = print_pdb(X, ab['binder_seq'], 'H')
                array2 = print_pdb(Y, ab['antigen_seq'], 'A')
                array = struc.array(array2 + array1)
                save_structure(path, array)
                try:
                    openmm_relax(path, use_gpu=True)
                    cands.append(path)
                    acc.append((ref, seq)) 
                except:
                    continue

            with multiprocessing.Pool(80) as pool:
                energy = pool.map(compute_energy, cands)

            for (ref, seq), e in zip(acc, energy):
                print(pdb, ref, seq, e)
