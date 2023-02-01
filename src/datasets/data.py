from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class CrystalData(Data):
    def __init__(self, *args, **kwargs):
        if "pos_cart" in kwargs:
            assert isinstance(kwargs["cell"], torch.FloatTensor)
            assert isinstance(kwargs["pos_cart"], torch.FloatTensor)
            assert isinstance(kwargs["num_atoms"], torch.LongTensor)

            cell = kwargs["cell"]
            pos_cart = kwargs["pos_cart"]
            num_atoms = kwargs["num_atoms"]

            batch = torch.arange(
                num_atoms.shape[0], dtype=torch.long, device=num_atoms.device
            ).repeat_interleave(num_atoms)

            pos = (
                torch.matmul(pos_cart.unsqueeze(1), torch.inverse(cell)[batch]).squeeze(
                    1
                )
                % 1
            )

            kwargs["pos"] = pos
            del kwargs["pos_cart"]

        super(CrystalData, self).__init__(*args, **kwargs)

        self._pos_cart = None

    @property
    def cell(self) -> torch.FloatTensor:
        return super(CrystalData, self).cell

    def set_cell(self, cell):
        self.cell = cell

        self._pos_cart = None

    @property
    def pos(self) -> torch.FloatTensor:
        return super(CrystalData, self).pos

    def set_pos(self, pos: torch.FloatTensor):
        self.pos = pos % 1

        self._pos_cart = None

    @property
    def device(self) -> torch.device:
        return self.cell.device

    @property
    def cell_lengths(self) -> torch.FloatTensor:
        return self.cell.norm(dim=2).t()

    @property
    def cell_angles(self) -> torch.FloatTensor:
        angles = torch.zeros_like(self.cell_lengths)

        i = torch.tensor([0, 1, 2], dtype=torch.long, device=self.device)
        j = torch.tensor([1, 2, 0], dtype=torch.long, device=self.device)
        k = torch.tensor([2, 0, 1], dtype=torch.long, device=self.device)

        cross = torch.cross(self.cell[:, j], self.cell[:, k], dim=2)
        dot = (self.cell[:, j] * self.cell[:, k]).sum(dim=2)

        angles[i, :] = torch.rad2deg(torch.atan2(cross.norm(dim=2), dot).t())

        inv_mask = (cross * self.cell[:, i]).sum(dim=2) < 0
        angles[inv_mask.t()] *= -1

        return angles

    @property
    def pos_cart(self) -> torch.FloatTensor:
        if self._pos_cart is None:
            self._pos_cart = torch.matmul(
                self.pos.unsqueeze(1), self.cell[self.batch]
            ).squeeze(1)

        return self._pos_cart

    def set_pos_cart(self, pos_cart: torch.FloatTensor, keep_inside: bool = True):
        pos = torch.matmul(
            pos_cart.unsqueeze(1), torch.inverse(self.cell)[self.batch]
        ).squeeze(1)
        if keep_inside:
            pos %= 1.0
        self.pos = pos

        self._pos_cart = None
