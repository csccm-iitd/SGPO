"""Natural Gradient Descent for variational GP parameters.

Natural gradient updates for the variational distribution are
*critical* for Sparse GP / VNN-GP convergence.  While GPyTorch
exposes ``gpytorch.optim.NGD``, combining it correctly with a
standard optimizer for the hyperparameters is surprisingly tricky.

This module provides a convenience wrapper:
- Variational parameters (q(u) mean & covariance) are updated with
  natural gradients at learning rate ``lr_ngd``.
- All other parameters (kernel, likelihood, NN) are updated with
  AdamW at learning rate ``lr``.

Reference:
    Hensman et al., "Scalable Variational GP Classification" (2015);
    Salimbeni et al., "Natural Gradients in Practice" (2018).
"""

import torch
import gpytorch


class NaturalGradientOptimizer:
    """Two-optimizer wrapper: NGD for variational params + AdamW for the rest.

    Usage::

        opt = NaturalGradientOptimizer(model, likelihood,
                                        lr=1e-2, lr_ngd=1e-1)
        for x, y in loader:
            opt.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            opt.step()

    Args:
        model: An ``ApproximateGP`` model.
        likelihood: GPyTorch likelihood.
        lr: Learning rate for hyperparameters (AdamW).
        lr_ngd: Learning rate for variational parameters (NGD).
        weight_decay: Weight decay for AdamW.
        betas: Adam betas for the hyperparameter optimizer.
    """

    def __init__(self, model, likelihood, lr=1e-2, lr_ngd=1e-1,
                 weight_decay=1e-4, betas=(0.9, 0.999)):

        # Separate variational from hyper/NN parameters
        variational_params = set()
        for name, param in model.named_parameters():
            if "variational" in name:
                variational_params.add(param)

        hyper_params = [p for p in model.parameters() if p not in variational_params]
        hyper_params += list(likelihood.parameters())

        self.ngd = gpytorch.optim.NGD(
            list(variational_params), num_data=1, lr=lr_ngd,
        )
        self.hyper = torch.optim.AdamW(
            hyper_params, lr=lr, weight_decay=weight_decay, betas=betas,
        )
        self._optimizers = [self.ngd, self.hyper]

    def set_num_data(self, n: int):
        """Update the NGD num_data (call before training if batch-size changes)."""
        for group in self.ngd.param_groups:
            group["num_data"] = n

    def zero_grad(self):
        for opt in self._optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self._optimizers:
            opt.step()

    def state_dict(self):
        return {
            "ngd": self.ngd.state_dict(),
            "hyper": self.hyper.state_dict(),
        }

    def load_state_dict(self, sd):
        self.ngd.load_state_dict(sd["ngd"])
        self.hyper.load_state_dict(sd["hyper"])
