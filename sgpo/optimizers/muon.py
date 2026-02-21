"""Muon optimizer 

- Significant wall-clock and sample-efficiency gains over AdamW on
  vision transformers and large-scale language models (NanoGPT,
  modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt).
- Natural fit for GP training where kernel matrices are inherently
  matrix-valued and orthogonality-preserving updates help.

Reference:
    Jordan et al., "Muon: An optimizer for hidden layers in neural
    networks" (2024).  https://arxiv.org/abs/2502.16982
"""

import torch
from torch.optim import Optimizer


def _newton_schulz_5(G: torch.Tensor, steps: int = 5,
                     eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix sign / polar factor via Newton-Schulz iteration.

    Maps G → U where G = U S V^T (SVD) — i.e. the orthogonal component.
    Uses 5th-order coefficients for fast convergence.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # 5th-order optimal coeffs
    X = G.float()
    # Normalise so spectral norm ≈ 1
    X = X / (X.norm(float("inf")) + eps)
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: Momentum with orthogonalised updates.

    Applies Newton-Schulz orthogonalisation to the momentum buffer,
    then uses the resulting direction for the parameter update.
    Only applies to parameters with ndim >= 2; scalar / 1-D
    parameters (biases, norms) fall back to standard SGD+momentum.

    Args:
        params: Iterable of parameters or param-groups.
        lr: Learning rate.
        momentum: Momentum coefficient (β).
        nesterov: Use Nesterov-style momentum.
        ns_steps: Number of Newton-Schulz iterations.
        weight_decay: Decoupled weight decay (AdamW-style).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                # Orthogonalise for matrix-shaped params
                if update.ndim >= 2:
                    original_shape = update.shape
                    # Reshape to 2-D if needed (e.g. Conv kernels)
                    if update.ndim > 2:
                        update = update.reshape(update.shape[0], -1)
                    update = _newton_schulz_5(update, steps=ns_steps)
                    update = update.reshape(original_shape)
                    # Scale by sqrt of dimensions (Muon prescription)
                    scale = max(1, update.shape[-2] / update.shape[-1]) ** 0.5
                    p.add_(update, alpha=-lr * scale)
                else:
                    # Fallback: plain SGD+momentum for 1-D / scalar
                    p.add_(update, alpha=-lr)

        return loss
