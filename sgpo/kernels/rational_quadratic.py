"""Rational Quadratic kernel — a scale mixture of RBF kernels.

    k(x, x') = \sigma^2 (1 + ||x-x'||² / (2\alpha l^2))^{-\alpha}

As alpha tends to infinity the RQ kernel converges to the RBF kernel. 
"""

import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel with learnable shape parameter alpha.

    Particularly useful for PDE problems where the solution possesses
    multi-scale spatial correlations (e.g. turbulence, shocks).

    Args:
        alpha_prior: Prior on alpha (shape).  Default unconstrained positive.
        ard_num_dims: Per-dimension lengthscale (ARD) if set.
        batch_shape: Batch shape for multi-output.
    """

    has_lengthscale = True

    def __init__(self, alpha_prior=None, ard_num_dims=None,
                 batch_shape=torch.Size([]), **kwargs):
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            **kwargs,
        )

        alpha_constraint = Positive()
        self.register_parameter(
            "raw_alpha",
            torch.nn.Parameter(torch.zeros(*batch_shape, 1)),
        )
        self.register_constraint("raw_alpha", alpha_constraint)

        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, self._alpha_param,
                                self._alpha_closure)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)

    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Scaled squared distance
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        if diag:
            diff = x1_ - x2_
            dist_sq = (diff ** 2).sum(dim=-1)
        else:
            dist_sq = self.covar_dist(x1_, x2_, square_dist=True)

        alpha = self.alpha
        # Broadcast alpha
        for _ in range(dist_sq.dim() - alpha.dim()):
            alpha = alpha.unsqueeze(-1)

        return (1.0 + dist_sq / (2.0 * alpha)) ** (-alpha)
