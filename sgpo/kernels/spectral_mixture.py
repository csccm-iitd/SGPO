""" Taken from Spectral Mixture kernel (Wilson & Adams, ICML 2013).

    k(τ) = \Sigma_q  w_q  exp(-2π²τ²v_q) cos(2πτμ_q)

where τ = |x - x'|, w_q are mixture weights, μ_q are spectral means,
and v_q are spectral variances.
"""

import math
import torch
import gpytorch
from gpytorch.kernels import Kernel
from torch.nn import Parameter


class SpectralMixtureKernel(Kernel):
    """Spectral Mixture (SM) kernel with *num_mixtures* Gaussian components.

    This kernel is particularly effective for:
    - Long-range correlations in PDE solutions
    - Discovering hidden periodicities in turbulence data
    - Extrapolation tasks in operator learning

    Args:
        num_mixtures: Number of Gaussian mixture components in the spectral
            density.  More mixtures → more expressive, but more parameters.
        ard_num_dims: If set, each input dimension gets separate spectral
            parameters (Automatic Relevance Determination).
        batch_shape: Batch shape for multi-output / LMC latent GPs.
    """

    has_lengthscale = False  # we parametrise via spectral params

    def __init__(self, num_mixtures: int = 4, ard_num_dims: int | None = None,
                 batch_shape: torch.Size = torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)

        self.num_mixtures = num_mixtures
        self.ard_num_dims = ard_num_dims

        # Dimension of spectral params (per-dim or shared)
        param_dim = ard_num_dims if ard_num_dims is not None else 1

        # Learnable spectral parameters (log-space for positivity)
        self.register_parameter(
            "raw_mixture_weights",
            Parameter(torch.zeros(*batch_shape, num_mixtures)),
        )
        self.register_parameter(
            "raw_mixture_means",
            Parameter(torch.randn(*batch_shape, num_mixtures, param_dim)),
        )
        self.register_parameter(
            "raw_mixture_scales",
            Parameter(torch.zeros(*batch_shape, num_mixtures, param_dim)),
        )

    @property
    def mixture_weights(self):
        return torch.softmax(self.raw_mixture_weights, dim=-1)

    @property
    def mixture_means(self):
        return self.raw_mixture_means

    @property
    def mixture_scales(self):
        return torch.nn.functional.softplus(self.raw_mixture_scales)

    def forward(self, x1, x2, diag=False, **params):
        # Pairwise differences: (..., n, m, d)
        diff = self.covar_dist(x1, x2, diag=diag, square_dist=False)

        if diag:
            # diff is (..., n)  → treat as 1-d
            diff = diff.unsqueeze(-1)  # (..., n, 1)

        # Reshape for broadcasting:  (batch, Q, 1, 1, D)  or  (batch, Q, 1, D) diag
        weights = self.mixture_weights  # (batch, Q)
        means = self.mixture_means      # (batch, Q, D)
        scales = self.mixture_scales    # (batch, Q, D)

        # Expand diff:  (..., n, m, D) → (..., 1, n, m, D)
        if not diag:
            tau = (x1.unsqueeze(-2) - x2.unsqueeze(-3))  # (..., n, m, D)
            tau = tau.unsqueeze(-4)  # (..., 1, n, m, D)
        else:
            tau = torch.zeros_like(x1).unsqueeze(-2)  # (..., n, 1, D)
            tau = tau.unsqueeze(-3)  # (..., 1, n, 1, D)

        # Expand spectral params
        # means:  (batch, Q, 1, 1, D)
        for _ in range(tau.dim() - means.dim()):
            means = means.unsqueeze(-2)
            scales = scales.unsqueeze(-2)

        exp_term = torch.exp(-2.0 * math.pi ** 2 * tau ** 2 * scales)
        cos_term = torch.cos(2.0 * math.pi * tau * means)
        # Product over dimensions, sum over mixtures
        component = (exp_term * cos_term).prod(dim=-1)  # (..., Q, n, m)

        # Weight and sum over Q
        weights_expanded = weights
        for _ in range(component.dim() - weights_expanded.dim()):
            weights_expanded = weights_expanded.unsqueeze(-1)
        result = (weights_expanded * component).sum(dim=-3)  # (..., n, m)

        if diag:
            result = result.squeeze(-1)

        return result

    def initialize_from_data(self, x, y=None):
        """Heuristic initialisation from training data (NYström-style)."""
        with torch.no_grad():
            # Estimate spectral density using FFT of empirical covariance
            x_flat = x.reshape(-1, x.shape[-1])
            n, d = x_flat.shape

            for q in range(self.num_mixtures):
                idx = torch.randint(0, n, (min(n, 1000),))
                subset = x_flat[idx]
                self.raw_mixture_means.data[..., q, :] = subset.mean(0) * (q + 1) / self.num_mixtures
                self.raw_mixture_scales.data[..., q, :] = torch.log(
                    torch.exp(subset.std(0) * (q + 1)) - 1 + 1e-6
                )
