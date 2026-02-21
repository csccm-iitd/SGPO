"""Basis-function mean functions for GP models.

"""

import math
import torch
import torch.nn as nn
import gpytorch


class FourierBasisMean(gpytorch.means.Mean):
    """Mean function as a truncated Fourier series.

        m(x) = \Sum_{k=1}^{K} [ a_k cos(2\pi kx/L) + b_k sin(2\pi kx/L) ] + c

    The period *L* and the coefficients *a_k*, *b_k*, *c* are all
    learnable parameters.

    Works for 1-D inputs.  For multi-D inputs each dimension
    is treated independently and the products are summed.

    Args:
        num_modes: Number of Fourier modes (K).
        period: Initial period length. Learnable unless ``learn_period=False``.
        learn_period: Whether to make the period a trainable parameter.
    """

    def __init__(self, num_modes: int = 8, period: float = 1.0,
                 learn_period: bool = True):
        super().__init__()
        self.num_modes = num_modes

        self.cos_coeffs = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.sin_coeffs = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

        if learn_period:
            self.raw_period = nn.Parameter(
                torch.tensor(math.log(math.exp(period) - 1.0))
            )
        else:
            self.register_buffer(
                "raw_period",
                torch.tensor(math.log(math.exp(period) - 1.0)),
            )

    @property
    def period(self):
        return torch.nn.functional.softplus(self.raw_period)

    def forward(self, x):
        # x: (..., d) — we use the last dim
        x_last = x[..., -1:]  # (..., 1)
        # k = 1..K
        k = torch.arange(1, self.num_modes + 1, device=x.device, dtype=x.dtype)
        # Phase:  (..., K)
        phase = 2.0 * math.pi * k * x_last / self.period
        # Linear combination
        result = (self.cos_coeffs * torch.cos(phase) +
                  self.sin_coeffs * torch.sin(phase)).sum(dim=-1)
        return result + self.bias


class PolynomialBasisMean(gpytorch.means.Mean):
    """Mean function as a polynomial in the input features.

    Args:
        input_dim: Input dimensionality.
        degree: Maximum polynomial degree.
    """

    def __init__(self, input_dim: int, degree: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        # One coefficient per (degree+1, input_dim) pair, plus a global bias
        self.coeffs = nn.Parameter(torch.zeros(degree + 1, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (..., d)
        # Evaluate each degree: x^0, x^1, ..., x^deg
        powers = []
        xp = torch.ones_like(x)
        for _ in range(self.degree + 1):
            powers.append(xp)
            xp = xp * x
        # Stack: (..., degree+1, d)
        poly = torch.stack(powers, dim=-2)
        # Weighted sum: (..., degree+1, d) × (degree+1, d) → (...)
        result = (poly * self.coeffs).sum(dim=(-2, -1))
        return result + self.bias
