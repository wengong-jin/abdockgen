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

import pdbfixer
import openmm
import biotite.structure as struc
from biotite.structure import AtomArray, Atom
from biotite.structure.io import save_structure
from biotite.structure.io.pdb import PDBFile

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


def openmm_relax(pdb_file, stiffness=10., tolerance=2.39, use_gpu=False):
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    if stiffness > 0:
        stiffness = stiffness * ENERGY / (LENGTH**2)
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        for residue in modeller.topology.residues():
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "C", "CB"]:
                    force.addParticle(
                            atom.index, modeller.positions[atom.index]
                    )
        system.addForce(force)

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


if __name__ == "__main__":
    model_ckpt, _, args = torch.load(sys.argv[1])
    model = RefineDocker(args).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()

    data = AntibodyComplexDataset(
            sys.argv[2],
            cdr_type=args.cdr,
            L_target=args.L_target,
    )

    with torch.no_grad():
        for ab in tqdm(data.data):
            pdb = ab['pdb']
            batch = make_batch([ab])
            batch[0][0].fill_(0)  # remove ground truth
            out = model(*batch)

            bind_X, _, bind_A, _ = batch[0]
            bind_mask = bind_A.clamp(max=1).float()

            X = out.bind_X[0].cpu().numpy()
            Y = np.array(ab['antigen_coords'])
            path = os.path.join('outputs', f'{pdb}_pred.pdb')
            array1 = print_pdb(X, ab['binder_seq'], 'H')
            array2 = print_pdb(Y, ab['antigen_seq'], 'A')
            array = struc.array(array2 + array1)
            save_structure(path, array)
            try:
                openmm_relax(path, use_gpu=True, stiffness=0)
            except:
                continue
