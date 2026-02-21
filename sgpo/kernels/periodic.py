"""Periodic kernel for signals with discoverable periodicity.

    k(x, x') = \sigma^2 exp( -2 sin^2(\pi|x-x'|/p) / l^2 )

where p is the period and ℓ the lengthscale.  
"""

import math
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class PeriodicKernel(Kernel):
    """Standard periodic (ExpSinSquared) kernel.

    Ideal for:
    - Wave equation solutions with known spatial period
    - Periodic boundary conditions in Navier-Stokes
    - Oscillatory forcing functions

    Args:
        period_length_prior: Prior on the period.
        ard_num_dims: Per-dimension parameters if set.
        batch_shape: Batch shape for multi-output.
    """

    has_lengthscale = True

    def __init__(self, period_length_prior=None, ard_num_dims=None,
                 batch_shape=torch.Size([]), **kwargs):
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            **kwargs,
        )

        period_constraint = Positive()
        param_dim = ard_num_dims if ard_num_dims is not None else 1
        self.register_parameter(
            "raw_period_length",
            torch.nn.Parameter(torch.zeros(*batch_shape, 1, param_dim)),
        )
        self.register_constraint("raw_period_length", period_constraint)

        if period_length_prior is not None:
            self.register_prior(
                "period_length_prior", period_length_prior,
                self._period_param, self._period_closure,
            )

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(
            raw_period_length=self.raw_period_length_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, **params):
        # Pairwise absolute differences
        if diag:
            diff = (x1 - x2)  # (..., n, d)
        else:
            diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3))  # (..., n, m, d)

        # sin²(π|Δ|/p) / ℓ², summed over dims
        period = self.period_length
        lengthscale = self.lengthscale.unsqueeze(-2) if not diag else self.lengthscale

        sin_term = torch.sin(math.pi * diff / period)
        dist = 2.0 * (sin_term / lengthscale) ** 2
        dist = dist.sum(dim=-1)

        return torch.exp(-dist)
