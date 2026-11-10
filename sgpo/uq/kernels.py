"""Kernels k(a, b; ell) used for K_a and K_x; the stationary ones have unit variance."""
import torch


def matern52(a, b, ell):
    r = torch.cdist(a, b) / ell
    s = 5 ** 0.5 * r
    return (1 + s + s ** 2 / 3) * torch.exp(-s)


def _stationary(f):
    return lambda a, b, ell: f(torch.cdist(a, b) / ell)


KERNELS = {
    "matern52": matern52,
    "matern32": _stationary(lambda r: (1 + 3 ** 0.5 * r) * torch.exp(-(3 ** 0.5) * r)),
    "matern12": _stationary(lambda r: torch.exp(-r)),
    "rbf": _stationary(lambda r: torch.exp(-0.5 * r ** 2)),
    "rq": _stationary(lambda r: (1 + r ** 2 / 2) ** -1.0),               # rational quadratic, alpha = 1
    "linear": lambda a, b, ell: a @ b.T / ell ** 2,                       # Bayesian linear regression on phi
}
