"""Optimizer builder — maps config strings to optimizer instances.

Centralises construction so YAML configs can specify any optimizer
by name (e.g. ``optimizer: ngd``) and the trainer will instantiate
the correct one.
"""

import torch
import gpytorch
from sgpo.optimizers.ngd import NaturalGradientOptimizer
from sgpo.optimizers.lbfgs import LBFGSOptimizer
from sgpo.optimizers.muon import Muon
from sgpo.optimizers.schedule_free import ScheduleFreeAdamW


OPTIMIZER_REGISTRY: dict[str, str] = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "ngd": "NaturalGradientOptimizer (NGD + AdamW)",
    "lbfgs": "LBFGSOptimizer",
    "muon": "Muon",
    "schedule_free": "ScheduleFreeAdamW",
}


def build_optimizer(
    optimizer_name: str,
    model,
    likelihood,
    *,
    lr: float = 1e-3,
    lr_ngd: float = 0.1,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    muon_momentum: float = 0.95,
    warmup_steps: int = 0,
    **extra,
):
    """Build an optimizer from a config string.

    Args:
        optimizer_name: Key in :data:`OPTIMIZER_REGISTRY`.
        model: GP model.
        likelihood: GPyTorch likelihood.
        lr: Learning rate.
        lr_ngd: NGD learning rate (only for ``'ngd'``).
        weight_decay: Weight decay.
        betas: Adam betas.
        muon_momentum: Momentum for Muon.
        warmup_steps: Warmup steps for ScheduleFree.

    Returns:
        An optimizer (or optimizer wrapper) ready for ``.zero_grad()``
        / ``.step()``.
    """
    params = list(model.parameters()) + list(likelihood.parameters())

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=betas)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                 betas=betas)

    if optimizer_name == "ngd":
        return NaturalGradientOptimizer(
            model, likelihood, lr=lr, lr_ngd=lr_ngd,
            weight_decay=weight_decay, betas=betas,
        )

    if optimizer_name == "lbfgs":
        return LBFGSOptimizer(params, lr=lr)

    if optimizer_name == "muon":
        return Muon(params, lr=lr, momentum=muon_momentum,
                    weight_decay=weight_decay)

    if optimizer_name == "schedule_free":
        return ScheduleFreeAdamW(
            params, lr=lr, betas=betas, weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )

    raise ValueError(
        f"Unknown optimizer: {optimizer_name!r}. "
        f"Available: {sorted(OPTIMIZER_REGISTRY)}"
    )
