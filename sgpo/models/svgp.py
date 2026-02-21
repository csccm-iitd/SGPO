import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    LMCVariationalStrategy,
)
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from sgpo.kernels.embedding_kernel import EmbeddingKernel


class SVGP(ApproximateGP):
    """Sparse Variational GP.

    Uses standard VariationalStrategy with learned inducing locations.
    Same kernel/mean/embedding interface as VNNGP for fair comparison.
    No nearest-neighbor approximation -- full variational inference.

    When a ``wno_embedding`` is supplied, the embedding is wrapped inside
    the kernel via :class:`EmbeddingKernel` so that internal GPyTorch
    calls (e.g. prior, KL) that access ``covar_module`` directly will
    also embed their inputs.

    Args:
        x_train: Training inputs (n, d), used to initialize inducing points.
        num_tasks: Number of output dimensions.
        num_latents: Number of independent latent GP.
        num_inducing: Number of inducing points.
        kernel_type: 'matern' or 'rbf'.
        kernel_nu: Smoothness for Matern kernel.
        use_ard: If True, use ARD.
        mean_module: Custom mean (e.g. WNOMean). None -> ConstantMean.
        wno_embedding: Feature extractor for kernel input. None -> identity.
    """

    def __init__(
        self,
        x_train,
        num_tasks,
        num_latents,
        num_inducing,
        kernel_type="matern",
        kernel_nu=1.5,
        use_ard=False,
        mean_module=None,
        wno_embedding=None,
    ):
        inducing_points = x_train[:num_inducing, :].clone()
        m = inducing_points.shape[0]

        variational_distribution = CholeskyVariationalDistribution(
            m, batch_shape=torch.Size([num_latents])
        )

        variational_strategy = LMCVariationalStrategy(
            VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)

        if mean_module is not None:
            self.mean_module = mean_module
        else:
            self.mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([num_latents])
            )

        batch_shape = torch.Size([num_latents])
        if use_ard:
            if wno_embedding is not None:
                ard_dims = wno_embedding.embed_dim
            else:
                ard_dims = num_tasks
        else:
            ard_dims = None

        if kernel_type == "matern":
            base_kernel = MaternKernel(
                nu=kernel_nu, batch_shape=batch_shape, ard_num_dims=ard_dims
            )
        elif kernel_type == "rbf":
            base_kernel = RBFKernel(
                batch_shape=batch_shape, ard_num_dims=ard_dims
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        inner_kernel = ScaleKernel(base_kernel)

        if wno_embedding is not None:
            self.covar_module = EmbeddingKernel(inner_kernel, wno_embedding)
        else:
            self.covar_module = inner_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return MultivariateNormal(mean_x, covar_x)
