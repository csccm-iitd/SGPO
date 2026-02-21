"""Schedule-Free AdamW (Defazio et al., 2024).

Reference:
    Defazio, Mishchenko, Khaled, Cutkosky,
    "The Road Less Scheduled" (2024).
    https://arxiv.org/abs/2405.15682
    Official repo: https://github.com/facebookresearch/schedule_free
"""

import math
import torch
from torch.optim import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """Schedule-Free AdamW.

    Maintains z (eval point) and x (model params) iterates.
    Call ``.train()`` before training and ``.eval()`` before
    evaluation/prediction to swap the model to the correct iterate.

    Args:
        params: Iterable of parameters.
        lr: Base learning rate.
        betas: Adam momentum coefficients.
        eps: Numerical stability.
        weight_decay: Decoupled weight decay.
        warmup_steps: Linear warmup steps.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps)
        super().__init__(params, defaults)
        self._step_count = 0
        self._is_train = True

    def _warmup_factor(self):
        ws = self.param_groups[0]["warmup_steps"]
        if ws <= 0:
            return 1.0
        return min(1.0, (self._step_count + 1) / ws)

    def eval(self):
        """Switch model params to the evaluation (z) iterate."""
        if not self._is_train:
            return
        for group in self.param_groups:
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if "z" not in state:
                    continue
                # p currently holds x; swap to z for eval
                state["x"] = p.data.clone()
                p.data.copy_(state["z"])
        self._is_train = False

    def train(self):
        """Switch model params back to the training (x) iterate."""
        if self._is_train:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "x" not in state:
                    continue
                p.data.copy_(state["x"])
        self._is_train = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr_scale = self._warmup_factor()
        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"] * lr_scale
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            ck = 1.0 / max(1, self._step_count)  # averaging coefficient

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["z"] = p.data.clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                z = state["z"]

                # Decoupled weight decay on z
                if wd != 0:
                    z.mul_(1.0 - lr * wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected second moment
                bc = math.sqrt(1 - beta2 ** self._step_count)
                denom = (exp_avg_sq.sqrt() / bc).add_(eps)

                # Update z
                z.addcdiv_(exp_avg, denom, value=-lr)

                # Update x (model params): interpolation
                # x = (1 - ck) * x + ck * z
                p.data.mul_(1 - ck).add_(z, alpha=ck)

        return loss
