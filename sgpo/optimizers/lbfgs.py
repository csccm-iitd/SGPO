"""L-BFGS wrapper for GP hyper-parameter optimisation.

Typical use-case: two-phase training
  1. Warm-start with Adam (fast, noisy updates for NN/WNO params).
  2. Fine-tune kernel + likelihood params with L-BFGS (precise).
"""

import torch


class LBFGSOptimizer:
    """L-BFGS optimizer with closure for GP training.

    Args:
        params: Iterable of parameters to optimise.
        lr: Learning rate (initial step size for line search).
        max_iter: Max L-BFGS iterations per ``.step()`` call.
        history_size: Number of past gradients stored.
        line_search_fn: ``'strong_wolfe'`` (recommended) or ``None``.
    """

    def __init__(self, params, lr=1.0, max_iter=20, history_size=50,
                 line_search_fn="strong_wolfe"):
        self.optimizer = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        self._last_loss = None

    def step(self, closure):
        """Run one L-BFGS step with the provided closure.

        Args:
            closure: Callable that computes loss, calls ``.backward()``,
                and returns the loss **value** (scalar tensor).
                Must be re-evaluable (called multiple times per step).

        Returns:
            float: The final loss value.
        """
        self._last_loss = self.optimizer.step(closure)
        return self._last_loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_epoch(self, model, likelihood, mll, x_train, y_train):
        """Convenience: run one L-BFGS "epoch" on the full dataset.

        This is useful when fine-tuning kernel/likelihood params after
        the main Adam training loop.

        Args:
            model: GP model (set to train-mode externally).
            likelihood: GPyTorch likelihood.
            mll: Marginal log-likelihood (e.g. VariationalELBO).
            x_train, y_train: Full training tensors.

        Returns:
            float: Final loss value.
        """
        def closure():
            self.optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            return loss

        return self.step(closure)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)
