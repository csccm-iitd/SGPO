import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.mlls import VariationalELBO

from sgpo.utils.logging import ExperimentLogger
from sgpo.utils.saving import save_results
from sgpo.utils.metrics import relative_l2_error
from sgpo.optimizers.builder import build_optimizer
from sgpo.optimizers.schedule_free import ScheduleFreeAdamW
from sgpo.optimizers.lbfgs import LBFGSOptimizer


class Trainer:
    """Unified training pipeline for VNNGP/SVGP models.

    Handles: training loop, batched prediction, evaluation, result saving.
    Supports all optimizers in the registry (adam, adamw, ngd, lbfgs,
    muon, schedule_free).
    """

    def __init__(self, model, likelihood, cfg, x_normalizer=None, y_normalizer=None):
        self.model = model
        self.likelihood = likelihood
        self.cfg = cfg
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.logger = ExperimentLogger(cfg)

        self.elbo_history = []
        self.mse_history = []

    def _build_optimizer(self):
        tcfg = self.cfg.training
        return build_optimizer(
            tcfg.optimizer,
            self.model,
            self.likelihood,
            lr=tcfg.lr,
            lr_ngd=getattr(tcfg, "lr_ngd", 0.1),
            weight_decay=tcfg.weight_decay,
            muon_momentum=getattr(tcfg, "muon_momentum", 0.95),
            warmup_steps=getattr(tcfg, "warmup_steps", 0),
        )

    def _build_scheduler(self):
        tcfg = self.cfg.training
        # No scheduler for optimizers that handle scheduling internally
        if tcfg.optimizer in ("schedule_free", "lbfgs"):
            return None
        # NGD returns a wrapper; schedule the inner hyper-optimizer
        opt = getattr(self.optimizer, "hyper", self.optimizer)
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=tcfg.scheduler_step, gamma=tcfg.scheduler_gamma
        )

    def _make_loader(self, x, y, shuffle=True):
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=shuffle)

    def train(self, x_tr, y_tr):
        self.model.train()
        self.likelihood.train()

        # Switch ScheduleFree to train mode
        if isinstance(self.optimizer, ScheduleFreeAdamW):
            self.optimizer.train()
        # Set NGD num_data
        if hasattr(self.optimizer, "set_num_data"):
            self.optimizer.set_num_data(y_tr.size(0))

        mll = VariationalELBO(self.likelihood, self.model, num_data=y_tr.size(0))
        mse_fn = torch.nn.MSELoss()

        # L-BFGS uses full-batch closure, not mini-batch loop
        if isinstance(self.optimizer, LBFGSOptimizer):
            self._train_lbfgs(mll, mse_fn, x_tr, y_tr)
            return

        self.logger.start_training()
        train_loader = self._make_loader(x_tr, y_tr)

        for epoch in range(self.cfg.training.num_epochs):
            self.logger.start_epoch()
            epoch_elbo = 0.0
            epoch_mse = 0.0

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = -mll(output, batch_y)
                loss_mse = mse_fn(output.mean, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_elbo += loss.item()
                epoch_mse += loss_mse.item()

            if self.scheduler is not None:
                self.scheduler.step()
            n_batches = len(train_loader)
            avg_elbo = epoch_elbo / n_batches
            avg_mse = epoch_mse / n_batches
            self.elbo_history.append(avg_elbo)
            self.mse_history.append(avg_mse)

            self.logger.log_epoch(epoch, {"elbo": avg_elbo, "mse": avg_mse})

    def _train_lbfgs(self, mll, mse_fn, x_tr, y_tr):
        """L-BFGS training: full-batch, closure-based."""
        self.logger.start_training()
        for epoch in range(self.cfg.training.num_epochs):
            self.logger.start_epoch()
            loss_val = self.optimizer.step_epoch(
                self.model, self.likelihood, mll, x_tr, y_tr,
            )
            with torch.no_grad():
                output = self.model(x_tr)
                mse_val = mse_fn(output.mean, y_tr).item()
            self.elbo_history.append(loss_val)
            self.mse_history.append(mse_val)
            self.logger.log_epoch(epoch, {"elbo": loss_val, "mse": mse_val})

    def predict(self, x_t):
        """Batched prediction. Returns mean_pred, var_pred on CPU."""
        self.model.eval()
        self.likelihood.eval()

        # ScheduleFree needs explicit eval call to swap iterates
        if isinstance(self.optimizer, ScheduleFreeAdamW):
            self.optimizer.eval()

        test_loader = self._make_loader(x_t, x_t, shuffle=False)  # dummy y
        all_means = []
        all_vars = []

        with torch.no_grad():
            for batch_x, _ in test_loader:
                output = self.model(batch_x)
                mean_batch = output.mean
                var_batch = output.variance

                if mean_batch.ndim == 3 and mean_batch.shape[1] == 1:
                    mean_batch = mean_batch.squeeze(1)
                    var_batch = var_batch.squeeze(1)

                all_means.append(mean_batch.cpu())
                all_vars.append(var_batch.cpu())

        mean_pred = torch.cat(all_means, dim=0)
        var_pred = torch.cat(all_vars, dim=0)

        # Decode if normalizer was applied during training
        if self.y_normalizer is not None:
            res = self.cfg.data.resolution
            std = self.y_normalizer.std + self.y_normalizer.eps
            if len(res) == 2:
                mean_pred = mean_pred.reshape(mean_pred.shape[0], res[0], res[1])
                mean_pred = self.y_normalizer.decode(mean_pred)
                mean_pred = mean_pred.reshape(mean_pred.shape[0], -1)

                # Variance transforms as var_y = std^2 * var_normalized
                var_pred = var_pred.reshape(var_pred.shape[0], res[0], res[1])
                var_pred = var_pred * (std ** 2)
                var_pred = var_pred.reshape(var_pred.shape[0], -1)
            else:
                mean_pred = self.y_normalizer.decode(mean_pred)
                var_pred = var_pred * (std ** 2)

        return mean_pred, var_pred

    def evaluate(self, mean_pred, y_t):
        """Compute and log evaluation metrics. Returns metrics dict."""
        y_t_cpu = y_t.detach().cpu()
        mse = torch.nn.functional.mse_loss(mean_pred, y_t_cpu).item()
        rel_err = relative_l2_error(mean_pred, y_t_cpu).item()

        metrics = {
            "mse": mse,
            "relative_l2_error": rel_err,
            "relative_l2_error_pct": 100.0 * rel_err,
        }
        self.logger.log_eval(metrics)
        return metrics

    def save(self, mean_pred, var_pred, y_t):
        """Save model, predictions, and loss history."""
        save_results(
            save_dir=self.cfg.logging.save_dir,
            config=self.cfg,
            model=self.model,
            mean_pred=mean_pred,
            var_pred=var_pred,
            y_test=y_t,
            elbo_history=self.elbo_history,
            mse_history=self.mse_history,
        )
        self.logger.finish()
