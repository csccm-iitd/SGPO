"""Gibbs (non-stationary) kernel 
"""

import torch
import torch.nn as nn
from gpytorch.kernels import Kernel


class _LengthscaleNet(nn.Module):
    """Small MLP that maps spatial location → positive lengthscale."""

    def __init__(self, input_dim, hidden_dim=64, min_ls=1e-3):
        super().__init__()
        self.min_ls = min_ls
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x) + self.min_ls


class GibbsKernel(Kernel):
    """Non-stationary Gibbs kernel with NN-parametrised lengthscale.

    The lengthscale varies across the input domain, making this kernel
    ideal for:
    - Shock / discontinuity capturing in Burgers' equation
    - Boundary-layer resolution in Navier-Stokes
    - Heterogeneous-media Darcy flow

    The lengthscale network is jointly trained with the GP parameters.

    Args:
        input_dim: Dimensionality of the input space.
        hidden_dim: Width of the lengthscale MLP.
        batch_shape: Batch shape for multi-output.
    """

    has_lengthscale = False  # lengthscale is input-dependent

    def __init__(self, input_dim, hidden_dim=64,
                 batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.input_dim = input_dim
        # One lengthscale net per batch element (or shared)
        if len(batch_shape) == 0:
            self.ls_net = _LengthscaleNet(input_dim, hidden_dim)
        else:
            # For batched GPs, share the lengthscale net across latents
            self.ls_net = _LengthscaleNet(input_dim, hidden_dim)

        self.register_parameter(
            "raw_outputscale",
            nn.Parameter(torch.zeros(*batch_shape, 1)),
        )

    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)

    def _lengthscale(self, x):
        """Evaluate lengthscale at locations x.  Returns (..., n, 1)."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        ls = self.ls_net(x_flat)
        return ls.reshape(*orig_shape[:-1], 1)

    def forward(self, x1, x2, diag=False, **params):
        ls1 = self._lengthscale(x1)  # (..., n, 1)
        ls2 = self._lengthscale(x2)  # (..., m, 1)

        ls1_sq = ls1 ** 2
        ls2_sq = ls2 ** 2
        d = x1.shape[-1]

        if diag:
            # (..., n, 1)
            sum_ls_sq = ls1_sq + ls2_sq  # (..., n, 1)
            diff = x1 - x2  # (..., n, d)
            dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
            prefactor = (2.0 * ls1 * ls2 / sum_ls_sq).pow(d / 2.0)
            result = prefactor * torch.exp(-dist_sq / sum_ls_sq)
            return (self.outputscale.unsqueeze(-1) * result).squeeze(-1)
        else:
            # (..., n, 1, 1) and (..., 1, m, 1) for broadcasting
            ls1_sq = ls1_sq.unsqueeze(-2)  # (..., n, 1, 1)
            ls2_sq = ls2_sq.unsqueeze(-3)  # (..., 1, m, 1)
            ls1_b = ls1.unsqueeze(-2)
            ls2_b = ls2.unsqueeze(-3)

            sum_ls_sq = ls1_sq + ls2_sq  # (..., n, m, 1)

            diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., n, m, d)
            dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (..., n, m, 1)

            prefactor = (2.0 * ls1_b * ls2_b / sum_ls_sq).pow(d / 2.0)
            result = prefactor * torch.exp(-dist_sq / sum_ls_sq)
            result = result.squeeze(-1)  # (..., n, m)

            outputscale = self.outputscale
            for _ in range(result.dim() - outputscale.dim()):
                outputscale = outputscale.unsqueeze(-1)
            return outputscale * result
