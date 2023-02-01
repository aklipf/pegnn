import torch
import torch.nn as nn

import tqdm

import os
import json
from dataclasses import dataclass


def save_step(spike_dir, batch, model, opti):
    os.makedirs(spike_dir, exist_ok=True)

    batch_dict = {
        "cell": batch.cell.tolist(),
        "pos": batch.pos.tolist(),
        "z": batch.z.tolist(),
        "num_atoms": batch.num_atoms.tolist(),
    }
    with open(os.path.join(spike_dir, "batch.json"), "w") as fp:
        json.dump(batch_dict, fp)

    grad_dict = {}
    for k, p in model.named_parameters():
        if p.grad is not None:
            grad_dict[k] = p.grad.tolist()

    with open(os.path.join(spike_dir, "grad.json"), "w") as fp:
        json.dump(grad_dict, fp)

    model_dict = {}
    for k, p in model.named_parameters():
        model_dict[k] = p.tolist()

    with open(os.path.join(spike_dir, "model.json"), "w") as fp:
        json.dump(model_dict, fp)

    torch.save(opti.state_dict(), os.path.join(spike_dir, "opti.pt"))


class LogSpike:
    def __init__(self, log_dir, threshold=0.5, verbose=False, debug=False):
        self.log_dir = log_dir
        self.prev_loss = None
        self.verbose = verbose
        self.debug = debug
        self.threshold = threshold

    def log(self, loss, opt_step, batch, model, opti):
        if self.prev_loss is not None:
            if abs((loss.item() / prev_loss) - 1.0) > self.threshold:

                if self.debug:
                    spike_dir = os.path.join(
                        self.log_dir,
                        "spike",
                        f"epoch_{opt_step}_loss_{loss.item():.3f}",
                    )
                    save_step(spike_dir, batch, model, opti)

                if self.verbose:
                    print(
                        f"loss spike detected (from {prev_loss:.6f} to {loss.item():.6f})"
                    )

        prev_loss = loss.item()


class AggregateMetrics:
    def __init__(self, writer, label):
        self.writer = writer
        self.label = label

        self.loss = []
        self.lengths_error = []
        self.angles_error = []

    def append(self, loss, metrics):
        self.loss.append(loss.item())
        self.lengths_error.append(metrics["lengths_error"].item())
        self.angles_error.append(metrics["angles_error"].item())

    def preview(self):
        return " ".join(
            [
                f"loss: {self.loss[-1]:.4f}",
                f"lengths error: {self.lengths_error[-1]:.4f}",
                f"angles error: {self.angles_error[-1]:.4f}",
            ]
        )

    def log(self, opt_step, clear=True, hparams=None):
        loss = torch.tensor(self.loss).mean().item()
        lengths_error = torch.tensor(self.lengths_error).mean().item()
        angles_error = torch.tensor(self.angles_error).mean().item()

        if self.writer is not None:
            self.writer.add_scalar(f"{self.label}/loss", loss, opt_step)
            self.writer.add_scalar(
                f"{self.label}/lengths_error", lengths_error, opt_step)
            self.writer.add_scalar(
                f"{self.label}/angles_error", angles_error, opt_step)

        metrics = {
            "loss": loss,
            "lengths_error": lengths_error,
            "angles_error": angles_error
        }

        if (self.writer is not None) and (hparams is not None):
            self.writer.add_hparams(hparams, metrics)

        if clear:
            self.loss = []
            self.lengths_error = []
            self.angles_error = []

        return metrics


def training_iterator(loader, total_step, verbose=True):
    def data_loop(dataloader, total):
        it = 0
        while it < total:
            for _, batch in zip(range(total - it), dataloader):
                yield batch
            it += len(dataloader)

    data_it = data_loop(loader, total_step + 1)

    if verbose:
        tqdm_bar = tqdm.tqdm(data_it, total=total_step)
        data_it = iter(tqdm_bar)
    else:
        tqdm_bar = None

    return enumerate(data_it), tqdm_bar


def validation_iterator(loader, verbose=True):
    if verbose:
        return tqdm.tqdm(loader, desc="validation", position=1, leave=False)

    return loader


def testing_iterator(loader, verbose=True):
    if verbose:
        return tqdm.tqdm(loader, desc="testing")

    return loader


class Checkpoints:
    def __init__(self, log_dir, model, opti):
        self.log_dir = log_dir

        self.opti = opti
        self.model = model

        torch.save(opti.state_dict(), os.path.join(log_dir, "opti.pt"))
        torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

        self.best = float("inf")
        self.filename_best = os.path.join(log_dir, "model.pt")

    def step(self, opt_step, metrics):
        metrics_sum = (
            metrics["lengths_error"]
            + metrics["angles_error"]
        )

        if metrics_sum < self.best:
            self.best = metrics_sum

            if "best" in self.filename_best:
                delete_file = self.filename_best
            else:
                delete_file = None

            backup_filename = (
                f"best_model_batch_{opt_step}_val_{self.best:.3f}".replace(
                    ".", "_")
            )
            self.filename_best = os.path.join(
                self.log_dir, backup_filename + ".pt")
            torch.save(self.model.state_dict(), self.filename_best)

            if delete_file is not None:
                os.remove(delete_file)

        torch.save(self.model.state_dict(),
                   os.path.join(self.log_dir, "model.pt"))
        torch.save(self.opti.state_dict(),
                   os.path.join(self.log_dir, "opti.pt"))

    def load_best(self):
        weights = torch.load(self.filename_best,
                             map_location=torch.device("cpu"))
        self.model.load_state_dict(weights)

        return self.model


@dataclass
class Hparams:
    batch_size: int = 1 << 8
    total_step: int = 1 << 15

    lr: float = 1e-4  # included every time in grid search
    beta1: float = 0.9
    grad_clipping: float = 1.0

    loss: str = "parameters_l1"

    knn: int = 16
    features: int = 128

    ops_config_type: str = "grad"
    ops_config_normalize: bool = True
    ops_config_edges: str = "n_ij"
    ops_config_triplets: str = "n_ij|n_ik|angle"

    mpnn_layers: int = 8
    steps: int = 4

    scale_limit_weights: float = 0.0  # included every time in grid search
    scale_hidden_dim: int = 256
    scale_layers: int = 1
    scale_limit_actions: float = 0.5  # included every time in grid search
    scale_reduce_rho: str = "mean"

    repeated: bool = False

    mlp_lattice: bool = False

    @property
    def ops_config(self):
        def split(s, delimiter):
            if len(s) > 0:
                return s.split(delimiter)
            return []

        return {
            "type": self.ops_config_type,
            "normalize": self.ops_config_normalize,
            "edges": split(self.ops_config_edges, "|"),
            "triplets": split(self.ops_config_triplets, "|"),
        }

    def from_json(self, file_name):
        with open(file_name, "r") as fp:
            hparams = json.load(fp)

        for key, value in hparams.items():
            assert key in self.__dict__

            self.__dict__[key] = value

    def to_json(self, file_name):
        with open(file_name, "w") as fp:
            json.dump(self.__dict__, fp, indent=4)

    def dict(self):
        return self.__dict__


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)
