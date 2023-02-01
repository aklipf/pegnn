import torch
import torch.nn as nn
import torch.nn.functional as F


from src.datasets.data import CrystalData
from src.utils.scaler import LatticeScaler
from src.models.operator.utils import lattice_params_to_matrix_torch

from typing import Dict, Tuple


def get_metrics(batch: CrystalData, reconstructed: torch.FloatTensor, scaler: LatticeScaler) -> Dict[str, torch.FloatTensor]:
    lengths_real, angles_real = scaler.get_lattices_parameters(batch.cell)

    if isinstance(reconstructed, tuple):
        lengths_denoised, angles_denoised = reconstructed
    else:
        lengths_denoised, angles_denoised = scaler.get_lattices_parameters(
            reconstructed)

    lengths_dist = torch.abs(lengths_denoised - lengths_real)
    angles_dist = torch.abs(angles_denoised - angles_real)

    return {
        "lengths_error": lengths_dist.mean().detach(),
        "angles_error": angles_dist.mean().detach()
    }


class LossLattice(nn.Module):
    def __init__(self, lattice_scaler: LatticeScaler):
        super().__init__()
        self.lattice_scaler = lattice_scaler

    def forward(self, batch: CrystalData, reconstructed: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


def get_loss(batch: CrystalData, model: nn.Module, loss_fn: LossLattice, return_batch: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    reconstructed = model(
        cell=batch.cell, x=batch.pos, z=batch.z, struct_size=batch.num_atoms
    )
    loss = loss_fn(batch, reconstructed)

    rec = reconstructed

    if isinstance(rec, tuple):
        rec = loss_fn.lattice_scaler.denormalise(
            rec[0], rec[1])

    metrics = get_metrics(batch, rec, loss_fn.lattice_scaler)

    if isinstance(rec, tuple):
        rec = lattice_params_to_matrix_torch(*rec)

    if return_batch:
        return loss, metrics, (batch.cell, rec, batch.pos, batch.z, batch.num_atoms)
    return loss, metrics


class LossLatticeParameters(LossLattice):
    def __init__(self, lattice_scaler: LatticeScaler, distance: str = "l1"):
        super().__init__(lattice_scaler=lattice_scaler)

        assert distance in ["l1", "mse"]

        self.distance = distance

    def forward(self, batch: CrystalData, reconstructed: torch.FloatTensor) -> torch.FloatTensor:
        param_real = self.lattice_scaler.normalise_lattice(batch.cell)

        if isinstance(reconstructed, tuple):
            param_reconstructed = reconstructed
        else:
            param_reconstructed = self.lattice_scaler.normalise_lattice(
                reconstructed)

        param_real = torch.cat(param_real, dim=1)
        param_reconstructed = torch.cat(param_reconstructed, dim=1)

        if self.distance == "l1":
            return F.l1_loss(param_reconstructed, param_real)
        elif self.distance == "mse":
            return F.mse_loss(param_reconstructed, param_real)
