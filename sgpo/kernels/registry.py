"""Kernel registry — single place to map config strings -> kernel objects.

Centralises kernel construction so that config YAML files can specify
any kernel by name (e.g. ``kernel: spectral_mixture``) and the model
builder will instantiate the correct class with the right arguments.
"""

import torch
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel

from sgpo.kernels.spectral_mixture import SpectralMixtureKernel
from sgpo.kernels.rational_quadratic import RationalQuadraticKernel
from sgpo.kernels.periodic import PeriodicKernel
from sgpo.kernels.gibbs import GibbsKernel
from sgpo.kernels.deep_kernel import DeepKernel


#: Maps config-string → builder function.
#: Each builder receives ``(batch_shape, ard_num_dims, **model_cfg)``
#: and returns a ready-to-use kernel.
KERNEL_REGISTRY: dict[str, type] = {
    "matern": MaternKernel,
    "rbf": RBFKernel,
    "spectral_mixture": SpectralMixtureKernel,
    "rational_quadratic": RationalQuadraticKernel,
    "periodic": PeriodicKernel,
    "gibbs": GibbsKernel,
    "deep_kernel": DeepKernel,
}


def build_kernel(
    kernel_type: str,
    batch_shape: torch.Size = torch.Size([]),
    ard_num_dims: int | None = None,
    *,
    kernel_nu: float = 1.5,
    num_mixtures: int = 4,
    input_dim: int | None = None,
    feature_extractor=None,
    base_kernel=None,
    **extra,
) -> "gpytorch.kernels.Kernel":
    """Build a kernel from a config string.

    Args:
        kernel_type: One of the keys in :data:`KERNEL_REGISTRY`.
        batch_shape: Batch shape (= num_latents for LMC models).
        ard_num_dims: None or int → per-dim lengthscales.
        kernel_nu: Matern smoothness (0.5, 1.5, 2.5).
        num_mixtures: Components for Spectral Mixture kernel.
        input_dim: Required for Gibbs kernel.
        feature_extractor: ``nn.Module`` for Deep Kernel.
        base_kernel: Base GPyTorch kernel for Deep Kernel.

    Returns:
        A GPyTorch ``Kernel`` instance wrapped in ``ScaleKernel``
        where appropriate.
    """
    if kernel_type not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel: {kernel_type!r}. "
            f"Available: {sorted(KERNEL_REGISTRY)}"
        )

    if kernel_type == "matern":
        base = MaternKernel(
            nu=kernel_nu, batch_shape=batch_shape, ard_num_dims=ard_num_dims,
        )
        return ScaleKernel(base, batch_shape=batch_shape)

    if kernel_type == "rbf":
        base = RBFKernel(
            batch_shape=batch_shape, ard_num_dims=ard_num_dims,
        )
        return ScaleKernel(base, batch_shape=batch_shape)

    if kernel_type == "spectral_mixture":
        return SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
        )

    if kernel_type == "rational_quadratic":
        base = RationalQuadraticKernel(
            ard_num_dims=ard_num_dims, batch_shape=batch_shape,
        )
        return ScaleKernel(base, batch_shape=batch_shape)

    if kernel_type == "periodic":
        base = PeriodicKernel(
            ard_num_dims=ard_num_dims, batch_shape=batch_shape,
        )
        return ScaleKernel(base, batch_shape=batch_shape)

    if kernel_type == "gibbs":
        if input_dim is None:
            raise ValueError("Gibbs kernel requires `input_dim`.")
        return GibbsKernel(input_dim=input_dim, batch_shape=batch_shape)

    if kernel_type == "deep_kernel":
        if feature_extractor is None or base_kernel is None:
            raise ValueError(
                "Deep kernel requires `feature_extractor` and `base_kernel`."
            )
        return DeepKernel(feature_extractor, base_kernel)

    # Fallback (should not reach here)
    raise ValueError(f"Unhandled kernel type: {kernel_type}")
