from ctypes import Structure
import torch
import torch.nn.functional as F

from ase.spacegroup import crystal
import ase.io as io

import pandas as pd

from src.utils.visualize import select

import os


def write_cif(file_name, idx, cell, pos, z, num_atoms):
    cell, pos, z = select(idx, cell, pos, z, num_atoms)
    c = crystal(z, basis=pos, cell=cell)
    c.write(file_name, format="cif")


def get_atoms(idx, cell, pos, z, num_atoms):
    cell, pos, z = select(idx, cell, pos, z, num_atoms)
    return crystal(z, basis=pos, cell=cell)


class AggregateBatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cell = []
        self.cell_noisy = []
        self.cell_denoised = []
        self.pos = []
        self.z = []
        self.num_atoms = []

    def append(self, cell, cell_denoised, pos, z, num_atoms):
        self.cell.append(cell.clone().detach().cpu())
        self.cell_denoised.append(cell_denoised.clone().detach().cpu())
        self.pos.append(pos.clone().detach().cpu())
        self.z.append(z.clone().detach().cpu())
        self.num_atoms.append(num_atoms.clone().detach().cpu())

    def cat(self):
        z = torch.cat(self.z, dim=0)

        if z.ndim == 1:
            z = F.one_hot(z, num_classes=100)
        return (
            torch.cat(self.cell, dim=0),
            torch.cat(self.cell_denoised, dim=0),
            torch.cat(self.pos, dim=0),
            z,
            torch.cat(self.num_atoms, dim=0)
        )

    def write(self, path, verbose=False):
        cell, cell_denoised, pos, z, num_atoms = self.cat()
        os.makedirs(path, exist_ok=True)

        iterator = range(cell.shape[0])

        if verbose:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc=f"saving cif to {path}")

        struct_original = []
        struct_denoised = []

        for idx in iterator:
            struct_original.append(get_atoms(idx, cell, pos, z, num_atoms))
            struct_denoised.append(
                get_atoms(idx, cell_denoised, pos, z, num_atoms))

        io.write(os.path.join(path, "original.cif"), struct_original)
        io.write(os.path.join(path, "generated.cif"), struct_denoised)
