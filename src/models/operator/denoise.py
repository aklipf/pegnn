import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.layers.operator.gnn as ops
from src.models.operator.utils import build_mlp, lattice_params_to_matrix_torch
from src.utils.geometry import Geometry

from torch_scatter import scatter_mean


class Denoise(nn.Module):
    def __init__(
        self,
        features: int,
        knn: int,
        ops_config: dict,
        mpnn: int,
        steps: int,
        scale_limit_weights: float,
        scale_hidden_dim: int,
        scale_layers: int,
        scale_limit_actions: float,
        scale_reduce_rho: str,
        repeated: bool,
        mlp_lattice: bool = False,
        lattice_scaler=None,
    ):
        super(Denoise, self).__init__()

        self.knn = knn
        self.steps = steps
        self.repeated = repeated
        self.mlp_lattice = mlp_lattice

        self.embedding = nn.Embedding(100, features)

        self.mpnn = nn.ModuleList(
            [ops.MPNN(features=features) for _ in range(mpnn)])

        self.I = nn.Parameter(torch.eye(3), requires_grad=False)

        if self.mlp_lattice:
            assert lattice_scaler is not None
            self.lattice_scaler = lattice_scaler
            self.lattice_pred = build_mlp(features, 128, 4, 6)
        elif self.repeated:
            self.update = ops.MPNN(features=features)

            self.actions = ops.Actions(
                features,
                knn,
                ops_config,
                scale_k=scale_limit_weights,
                hidden_dim=scale_hidden_dim,
                n_layers=scale_layers,
                limit_actions=scale_limit_actions,
                reduce_rho=scale_reduce_rho,
            )
        else:
            self.update = nn.ModuleList(
                [ops.MPNN(features=features) for _ in range(self.steps)]
            )

            self.actions = nn.ModuleList(
                [
                    ops.Actions(
                        features,
                        knn,
                        ops_config,
                        scale_k=scale_limit_weights,
                        hidden_dim=scale_hidden_dim,
                        n_layers=scale_layers,
                        limit_actions=scale_limit_actions,
                        reduce_rho=scale_reduce_rho,
                    )
                    for _ in range(self.steps)
                ]
            )
        self.it = 0

    def actions_init(self, cell: torch.FloatTensor) -> torch.FloatTensor:
        return self.I.unsqueeze(0).repeat(cell.shape[0], 1, 1)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        struct_size: torch.FloatTensor,
        edge_index: torch.LongTensor = None,
        edge_attr: torch.LongTensor = None,
        step: int = None,
    ):
        geometry = Geometry(cell, struct_size, x % 1, knn=self.knn,
                            edge_index=edge_index, edge_attr=edge_attr)

        if step is None:
            step = self.steps

        h = self.embedding(z)

        for l in self.mpnn:
            h = l(geometry, h)

        if self.mlp_lattice:
            latent = scatter_mean(h, geometry.batch, dim=0)

            lattice = self.lattice_pred(latent)

            lengths, angles = self.lattice_scaler.denormalise(
                lattice[:, :3], lattice[:, 3:]
            )

            cell_prime = lattice_params_to_matrix_torch(lengths, angles)

            action = torch.bmm(cell_prime, torch.inverse(cell))

            return cell_prime, [cell_prime], [action]
        else:
            action_rho = self.actions_init(cell)

            rho_list = []
            actions_list = []

            if self.repeated:
                actions = self.actions
                update = self.update

            for i in range(step):
                if not self.repeated:
                    actions = self.actions[i]
                    update = self.update[i]

                h = update(geometry, h)
                edges_weights, triplets_weights = actions(geometry, h)

                rho_prime, action = actions.apply(
                    geometry, edges_weights, triplets_weights
                )
                action_rho = torch.bmm(action, action_rho)
                rho_prime = torch.bmm(action_rho, cell)

                rho_list.append(rho_prime)
                actions_list.append(action_rho)

                geometry.rho = rho_prime
                geometry.update_vectors()

            return geometry.rho, rho_list, actions_list
