import torch
import torch.nn as nn


class RandomMatrixSL3Z(nn.Module):
    def __init__(self):
        super().__init__()

        generators = torch.tensor(
            [
                [[1, 0, 1], [0, -1, -1], [0, 1, 0]],
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                [[0, 1, 0], [1, 0, 0], [-1, -1, -1]],
            ],
            dtype=torch.float32,
        )
        generators = torch.cat((generators, torch.inverse(generators)), dim=0)

        self.generators = nn.Parameter(generators, requires_grad=False)

    @property
    def device(self):
        return self.generators.device

    def forward(self, batch_size, e=5):
        n = 1 << e
        g_idx = torch.randint(
            0, self.generators.shape[0], (batch_size * n,), device=self.device
        )
        M = self.generators[g_idx]

        for _ in range(e):
            M = M.view(2, -1, 3, 3)
            M = torch.bmm(M[0], M[1])
        return torch.round(M)

def apply_sl3z(g, rho, x, batch):
    rho_prime = torch.bmm(rho, torch.inverse(g))
    x_prime = (torch.bmm(g[batch], x.unsqueeze(2)) % 1).squeeze(2)

    return rho_prime, x_prime

class RandomSLZ(nn.Module):
    def __init__(self):
        super().__init__()

        self.generator = RandomMatrixSL3Z()

    def forward(self, rho, x, batch):
        g = self.generator(rho.shape[0])

        rho_prime = torch.bmm(rho, g)
        x_prime = (torch.bmm(torch.inverse(g)[batch], x.unsqueeze(2)) % 1).squeeze(2)

        return rho_prime, x_prime
