"""Optimizers for GP training.

Includes Natural Gradient Descent (essential for variational GPs),
L-BFGS (second-order method), and modern alternatives that have
gained significant traction for high-dimensional non-convex problems.
"""

from sgpo.optimizers.ngd import NaturalGradientOptimizer
from sgpo.optimizers.lbfgs import LBFGSOptimizer
from sgpo.optimizers.muon import Muon
from sgpo.optimizers.schedule_free import ScheduleFreeAdamW
from sgpo.optimizers.builder import build_optimizer, OPTIMIZER_REGISTRY

__all__ = [
    "NaturalGradientOptimizer",
    "LBFGSOptimizer",
    "Muon",
    "ScheduleFreeAdamW",
    "build_optimizer",
    "OPTIMIZER_REGISTRY",
]
