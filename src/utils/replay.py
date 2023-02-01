import torch


class Replay:
    def __init__(self, batch_size: int, max_depth: int = 32, proba_in: float = 0.1):
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.proba_in = proba_in

        self.cell = torch.zeros(0, 3, 3, dtype=torch.float32)
        self.pos = torch.zeros(0, 3, dtype=torch.float32)
        self.z = torch.zeros(0, dtype=torch.float32)
        self.num_atoms = torch.zeros(0, dtype=torch.long)
        self.depth = 0

    def push(self, cell, pos, z, num_atoms):
        if torch.rand(1) < self.proba_in:
            cell = cell.clone().detach().cpu()
            pos = pos.clone().detach().cpu()
            z = z.clone().detach().cpu()
            num_atoms = num_atoms.clone().detach().cpu()

            if self.depth < self.max_depth:
                self.cell = torch.cat((self.cell, cell))
                self.pos = torch.cat((self.pos, pos))
                self.z = torch.cat((self.z, z))
                self.num_atoms = torch.cat((self.num_atoms, num_atoms))
                self.depth += 1
            else:
                struct_idx = torch.arange(
                    self.num_atoms.shape[0], device=self.num_atoms.device
                )
                batch = struct_idx.repeat_interleave(self.num_atoms)

                remove = torch.randint(self.depth, (1,)) * self.batch_size
                mask = (batch < remove) | ((remove + self.batch_size) <= batch)

                self.cell = torch.cat(
                    (self.cell[:remove], self.cell[(remove + self.batch_size) :], cell)
                )
                self.pos = torch.cat((self.pos[mask], pos))
                self.z = torch.cat((self.z[mask], z))
                self.num_atoms = torch.cat(
                    (
                        self.num_atoms[:remove],
                        self.num_atoms[(remove + self.batch_size) :],
                        num_atoms,
                    )
                )

    def random(self, device="cpu"):
        assert self.num_atoms.shape[0] > 0

        struct_idx = torch.arange(self.num_atoms.shape[0], device=self.num_atoms.device)
        batch = struct_idx.repeat_interleave(self.num_atoms)

        idx = torch.randperm(self.num_atoms.shape[0])[: self.batch_size]

        mask = (batch[:, None] == idx[None, :]).any(dim=1)

        return (
            self.cell[idx].to(device),
            self.pos[mask].to(device),
            self.z[mask].to(device),
            self.num_atoms[idx].to(device),
        )
