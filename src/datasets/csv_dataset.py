from typing import Iterator
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

import torch
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import neighbor_list
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .data import CrystalData
from src.models.layers.random import RandomMatrixSL3Z, apply_sl3z

import multiprocessing as mp
import warnings
import os
import json


def process_cif(args):
    (cif, warning_queue) = args

    with warnings.catch_warnings(record=True) as ws:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        struct = Structure.from_str(cif, fmt="cif")

        if warning_queue is not None:
            for w in ws:
                warning_queue.put((hash(str(w.message)), w))

    lengths = np.array(struct.lattice.abc, dtype=np.float32)
    angles = np.array(struct.lattice.angles, dtype=np.float32)

    atoms = AseAtomsAdaptor.get_atoms(struct)

    atoms.set_scaled_positions(atoms.get_scaled_positions(wrap=True))

    assert (0 <= atoms.get_scaled_positions()).all() and (
        atoms.get_scaled_positions() < 1).all()

    cell = atoms.cell.array.astype(np.float32)
    z = np.array(struct.atomic_numbers, dtype=np.long)
    pos = struct.frac_coords.astype(np.float32)

    data = {
        "lattice": cell,
        "lengths": lengths,
        "angles": angles,
        "z": z,
        "pos": pos
    }

    return data, np.unique(z), pos.shape[0]


class CSVDataset(InMemoryDataset):
    def __init__(self, csv_file: str, warn: bool = False, multithread: bool = True, verbose: bool = True, noise_scale: float = 0.1, knn: float = 8, sl3z_aug: bool = False):
        super().__init__()

        self._raw_file_names = [csv_file]
        df = pd.read_csv(csv_file)

        if warn:
            m = mp.Manager()
            warning_queue = m.Queue()
        else:
            warning_queue = None

        iterator = [(row["cif"], warning_queue)
                    for _, row in df.iterrows()]

        if multithread:
            if verbose:
                result = process_map(
                    process_cif, iterator, desc=f"loading dataset {csv_file}", chunksize=8)
            else:
                with mp.Pool(mp.cpu_count()) as p:
                    result = p.map(process_cif, iterator)
        else:
            result = []

            if verbose:
                iterator = tqdm(
                    iterator, desc=f"loading dataset {csv_file}", total=len(df))

            for args in iterator:
                result.append(process_cif(args))

        if warn:
            warnings_type = {}
            while not warning_queue.empty():
                key, warning = warning_queue.get()
                if key not in warnings_type:
                    warnings_type[key] = warning

            for w in warnings_type.values():
                warnings.warn_explicit(
                    w.message, category=w.category, filename=w.filename, lineno=w.lineno
                )

        self._elements = set(
            np.unique(np.concatenate([z for _, z, _ in result])))

        size = np.array([s for _, _, s in result])
        max_size = np.max(size)
        min_size = np.min(size)

        self.data = [c for c, _, _ in result]

        if verbose:
            print(
                f"dataset statistics: count={len(self.data)}, min={min_size}, max={max_size}")

    @ property
    def raw_file_names(self):
        return self._raw_file_names

    @ property
    def processed_file_names(self):
        return []

    def len(self) -> int:
        return len(self.data)

    def get_sample_size(self, idx: int) -> int:
        return len(self.data[idx]["z"])

    def get(self, idx: int) -> Data:
        lattice = torch.from_numpy(self.data[idx]["lattice"]).unsqueeze(0)
        z = torch.from_numpy(self.data[idx]["z"])
        pos = torch.from_numpy(self.data[idx]["pos"])

        return CrystalData(
            z=z,
            pos=pos,
            cell=lattice,
            num_atoms=z.shape[0]
        )
