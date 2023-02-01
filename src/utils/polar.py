import torch
import unittest

__all__ = ["polar"]


def polar(a: torch.FloatTensor, side: str = "right"):
    if side not in ["right", "left"]:
        raise ValueError("`side` must be either 'right' or 'left'")

    assert a.ndim == 3 and a.shape[1] == a.shape[2]

    w, s, vh = torch.linalg.svd(a, full_matrices=False)

    u = torch.bmm(w, vh)
    if side == "right":
        # a = up
        p = torch.bmm(torch.transpose(vh, 1, 2).conj() * s[:, None], vh)
    else:
        # a = pu
        p = torch.bmm(w * s[:, None], torch.transpose(w, 1, 2).conj())

    mask = torch.where(torch.det(u) < 0, -1.0, 1.0)

    u *= mask[:, None, None]
    p *= mask[:, None, None]

    return u, p


class TestPolar(unittest.TestCase):
    def test_left(self):
        import numpy as np
        from scipy.linalg import polar as polat_gt

        torch.manual_seed(0)

        A = torch.matrix_exp(torch.randn(1 << 10, 3, 3))

        R, K = polar(A, side="left")

        for i in range(A.shape[0]):
            R_gt, K_gt = polat_gt(A[i].numpy(), side="left")
            R_gt = torch.from_numpy(R_gt)
            K_gt = torch.from_numpy(K_gt)
            if torch.det(R_gt) < 0:
                R_gt -= R_gt
                K_gt -= K_gt

            self.assertAlmostEqual((R[i] - R_gt).abs().sum().item(), 0.0, places=4)
            self.assertAlmostEqual((K[i] - K_gt).abs().sum().item(), 0.0, places=4)

    def test_right(self):
        import numpy as np
        from scipy.linalg import polar as polat_gt

        torch.manual_seed(0)

        A = torch.matrix_exp(torch.randn(1 << 10, 3, 3))

        R, K = polar(A, side="right")

        for i in range(A.shape[0]):
            R_gt, K_gt = polat_gt(A[i].numpy(), side="right")
            R_gt = torch.from_numpy(R_gt)
            K_gt = torch.from_numpy(K_gt)
            if torch.det(R_gt) < 0:
                R_gt -= R_gt
                K_gt -= K_gt

            self.assertAlmostEqual((R[i] - R_gt).abs().sum().item(), 0.0, places=4)
            self.assertAlmostEqual((K[i] - K_gt).abs().sum().item(), 0.0, places=4)


def volume(x: torch.FloatTensor) -> torch.FloatTensor:
    return (torch.cross(x[:, :, 0], x[:, :, 1]) * x[:, :, 2]).sum(dim=1).abs()


def volume2(x):
    return torch.linalg.svd(x)[1].prod(dim=1).abs().detach()


if __name__ == "__main__":

    from torch import tensor

    calc_scale = 1.5
    rho = tensor(
        [
            [-1.9330e-01, 3.3560e00, -2.1579e00],
            [6.8199e01, -3.8512e02, 2.6373e02],
            [-3.6272e01, 2.0426e02, -1.3885e02],
        ]
    )
    actions_rho = tensor(
        [
            [0.9919, 1.0756, -0.5697],
            [1.0756, -143.5471, 76.5437],
            [-0.5697, 76.5437, -39.5333],
        ]
    )
    action_normalize = tensor(
        [
            [0.1996, 0.2165, -0.1146],
            [0.2165, -28.8877, 15.4038],
            [-0.1146, 15.4038, -7.9557],
        ]
    )

    rho.unsqueeze_(0)
    actions_rho.unsqueeze_(0)
    action_normalize.unsqueeze_(0)

    """
    print(
        torch.cross(
            tensor([-1.9330e-01, 6.8199e01, -3.6272e01]),
            tensor([3.3560e00, -3.8512e02, 2.0426e02]),
        )
        .dot(tensor([-2.1579e00, 2.6373e02, -1.3885e02]))
        .abs()
    )
    print(volume2(rho))
    print(volume(rho))
    exit()
    """

    print(torch.linalg.matrix_rank(rho))
    print(torch.linalg.matrix_rank(actions_rho, hermitian=True))
    U = torch.linalg.svd(rho).U[0]
    print(
        torch.dot(U[:, 0], U[:, 1]),
        torch.dot(U[:, 1], U[:, 2]),
        torch.dot(U[:, 2], U[:, 0]),
    )
    U = torch.linalg.svd(actions_rho).U[0]
    print(
        torch.dot(U[:, 0], U[:, 1]),
        torch.dot(U[:, 1], U[:, 2]),
        torch.dot(U[:, 2], U[:, 0]),
    )
    print(volume(action_normalize) * volume(rho))
    print(volume(torch.bmm(action_normalize, rho)))
    print(volume2(torch.bmm(action_normalize, rho)))

    # unittest.main()
