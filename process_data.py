import csv
import sys
import os
import numpy as np
import json
import argparse
from prody import *
from sidechainnet.utils.measure import *
from tqdm import tqdm

def tocdr(resseq):
    if 27 <= resseq <= 38:
        return '1'
    elif 56 <= resseq <= 65:
        return '2'
    elif 105 <= resseq <= 117:
        return '3'
    else:
        return '0'


if __name__ == "__main__":
    pdb_id, hchain, achain = sys.argv[1:4]

    hchain = parsePDB(pdb_id, model=1, chain=hchain)
    _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
    hcdr = ''.join([tocdr(res.getResnum()) for res in hchain.iterResidues()])
    hcdr = hcdr[:len(hseq)]
    hcoords = hcoords.reshape((len(hseq), 14, 3))
    hcoords = eval(np.array2string(hcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))

    achain = parsePDB(pdb_id, model=1, chain=achain)
    _, acoords, aseq, _, _ = get_seq_coords_and_angles(achain)
    acoords = acoords.reshape((len(aseq), 14, 3))
    acoords = eval(np.array2string(acoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))

    s = json.dumps({
        "pdb": pdb_id, 
        "antibody_seq": hseq, "antibody_cdr": hcdr, "antibody_coords": hcoords,
        "antigen_seq": aseq, "antigen_coords": acoords, 
    })
    print(s)
