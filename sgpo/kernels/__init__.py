"""Custom GP kernels for PDE operator learning.

Provides stationary and non-stationary kernel variants that are
commonly needed in physical-system modelling but absent from GPyTorch.
"""

from sgpo.kernels.spectral_mixture import SpectralMixtureKernel
from sgpo.kernels.rational_quadratic import RationalQuadraticKernel
from sgpo.kernels.periodic import PeriodicKernel
from sgpo.kernels.gibbs import GibbsKernel
from sgpo.kernels.deep_kernel import DeepKernel
from sgpo.kernels.registry import KERNEL_REGISTRY, build_kernel

__all__ = [
    "SpectralMixtureKernel",
    "RationalQuadraticKernel",
    "PeriodicKernel",
    "GibbsKernel",
    "DeepKernel",
    "KERNEL_REGISTRY",
    "build_kernel",
]
