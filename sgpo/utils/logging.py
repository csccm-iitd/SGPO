import time
from dataclasses import asdict


class ExperimentLogger:
    """Console + optional wandb logging with timing."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_wandb = cfg.logging.use_wandb
        self._wandb_run = None
        self._train_start = None
        self._epoch_start = None
        self._epoch_times = []
        self._best_elbo = float("inf")
        self._best_epoch = 0

        if self.use_wandb:
            import wandb
            run_name = cfg.logging.wandb_run_name or f"{cfg.data.name}_{cfg.model.model_type}"
            self._wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                name=run_name,
                config=asdict(cfg),
            )

    def log_header(self, device, n_params, n_params_lik, data_shapes):
        """Print experiment header at the start of training."""
        print("=" * 70)
        print(f"  Experiment : {self.cfg.data.name}")
        print(f"  Model      : {self.cfg.model.model_type} | Kernel: {self.cfg.model.kernel}")
        print(f"  Mean       : {self.cfg.model.mean_type}")
        print(f"  Device     : {device}")
        print(f"  Params     : model={n_params:,} | likelihood={n_params_lik:,}")
        print(f"  Data       : {data_shapes}")
        print(f"  Optimizer  : {self.cfg.training.optimizer} | LR: {self.cfg.training.lr}")
        print(f"  Epochs     : {self.cfg.training.num_epochs} | Batch: {self.cfg.training.batch_size}")
        print("=" * 70)

    def start_training(self):
        """Call at the start of training to begin the global timer."""
        self._train_start = time.time()

    def start_epoch(self):
        """Call at the start of each epoch to begin epoch timer."""
        self._epoch_start = time.time()

    def log_epoch(self, epoch, metrics):
        """Log metrics for an epoch with timing."""
        epoch_time = 0.0
        if self._epoch_start is not None:
            epoch_time = time.time() - self._epoch_start
            self._epoch_times.append(epoch_time)

        # Track best ELBO
        elbo = metrics.get("elbo", float("inf"))
        if elbo < self._best_elbo:
            self._best_elbo = elbo
            self._best_epoch = epoch + 1

        if self.use_wandb:
            import wandb
            wandb.log({**metrics, "epoch_time": epoch_time}, step=epoch)

        if (epoch + 1) % self.cfg.logging.log_interval == 0:
            parts = [f"Epoch {epoch + 1:>4d}/{self.cfg.training.num_epochs}"]
            for k, v in metrics.items():
                parts.append(f"{k}: {v:.4f}")
            parts.append(f"time: {epoch_time:.1f}s")
            print(" | ".join(parts))

    def log_eval(self, metrics):
        """Log final evaluation metrics."""
        print("-" * 50)
        print("  Evaluation Results:")
        if self.use_wandb:
            import wandb
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        for k, v in metrics.items():
            print(f"    {k}: {v:.6f}")
        print("-" * 50)

    def log_summary(self):
        """Print final training summary."""
        total_time = time.time() - self._train_start if self._train_start else 0.0
        avg_epoch = sum(self._epoch_times) / len(self._epoch_times) if self._epoch_times else 0.0
        print("=" * 70)
        print("  Training Summary:")
        print(f"    Total time     : {total_time:.1f}s ({total_time / 60:.1f}min)")
        print(f"    Avg epoch time : {avg_epoch:.2f}s")
        print(f"    Best ELBO      : {self._best_elbo:.4f} (epoch {self._best_epoch})")
        print(f"    Total epochs   : {len(self._epoch_times)}")
        print("=" * 70)

    def finish(self):
        self.log_summary()
        if self.use_wandb and self._wandb_run is not None:
            self._wandb_run.finish()
