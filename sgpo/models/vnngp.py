import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    MeanFieldVariationalDistribution,
    LMCVariationalStrategy,
)
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from sgpo.kernels.embedding_kernel import EmbeddingKernel


class VNNGP(ApproximateGP):


    def __init__(
        self,
        x_train,
        num_tasks,
        num_latents,
        num_nn,
        num_inducing,
        training_batch_size,
        kernel_type="matern",
        kernel_nu=2.5,
        use_ard=False,
        mean_module=None,
        wno_embedding=None,
    ):
        inducing_points = x_train[:num_inducing, :].clone().contiguous()
        m = inducing_points.shape[0]

        variational_distribution = MeanFieldVariationalDistribution(
            m, batch_shape=torch.Size([num_latents])
        )

        variational_strategy = LMCVariationalStrategy(
            NNVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                k=num_nn,
                training_batch_size=training_batch_size,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)

        # Mean function
        if mean_module is not None:
            self.mean_module = mean_module
        else:
            self.mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([num_latents])
            )

        # Kernel: ScaleKernel(BaseKernel), optionally wrapped with embedding
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
            # Wrap embedding inside the kernel so that NNVariationalStrategy's
            # direct covar_module calls also embed their inputs.
            self.covar_module = EmbeddingKernel(inner_kernel, wno_embedding)
        else:
            self.covar_module = inner_kernel

    def forward(self, x):
        # The kernel handles embedding internally (via EmbeddingKernel).
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return MultivariateNormal(mean_x, covar_x)

    # ------------------------------------------------------------------
    # Device transfer fix: NNVariationalStrategy stores nn_xinduce_idx
    # as a plain Python attribute (not a buffer), so the default .to()
    # / .cuda() calls don't move it.  We handle that here.
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        nn_strat = self.variational_strategy.base_variational_strategy
        if hasattr(nn_strat, "nn_xinduce_idx"):
            target = nn_strat.inducing_points.device
            if nn_strat.nn_xinduce_idx.device != target:
                nn_strat.nn_xinduce_idx = nn_strat.nn_xinduce_idx.to(target)
        return result
