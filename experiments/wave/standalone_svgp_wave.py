#!/usr/bin/env python
"""Standalone Wave experiment — SVGP (no WNO).

    u(x, t=0) -> u(x, t=80)

Usage:
    python experiments/wave/standalone_svgp_wave.py
"""

import sys, os, time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import gpytorch
from gpytorch.mlls import VariationalELBO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sgpo.models.svgp import SVGP
from sgpo.utils.metrics import relative_l2_error
from sgpo.utils.misc import set_seed, resolve_device, count_params

# ── Hyperparameters ──────────────────────────────────────────────────

TRAIN_PATH = "/DATA/Sawan_projects/research/DATA/train_IC2.npz"
TEST_PATH = "/DATA/Sawan_projects/research/DATA/test_IC2.npz"
NTRAIN, NTEST = 1000, 100
T_IN, T_OUT = 0, 80

NUM_LATENTS = 60
NUM_INDUCING = 200
KERNEL = "matern"
KERNEL_NU = 2.5

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 1000
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.75

SEED = 0
DEVICE = "cuda:0"
SAVE_DIR = "results/wave"


def main():
    set_seed(SEED)
    device = resolve_device(DEVICE)

    print("=" * 70)
    print("  Wave — SVGP (plain, no WNO)")
    print(f"  Device: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print("=" * 70)

    u_train = np.load(TRAIN_PATH)["u"]
    u_test = np.load(TEST_PATH)["u"]

    x_train = torch.tensor(u_train[:NTRAIN, T_IN, :]).float()
    y_train = torch.tensor(u_train[:NTRAIN, T_OUT, :]).float()
    x_test = torch.tensor(u_test[:NTEST, T_IN, :]).float()
    y_test = torch.tensor(u_test[:NTEST, T_OUT, :]).float()
    print(f"  x_train: {list(x_train.shape)}, y_train: {list(y_train.shape)}")

    x_tr = x_train.reshape(NTRAIN, -1).to(device)
    y_tr = y_train.reshape(NTRAIN, -1).to(device)
    x_t = x_test.reshape(NTEST, -1).to(device)
    y_t = y_test.reshape(NTEST, -1).to(device)

    num_tasks = x_tr.shape[-1]

    model = SVGP(
        x_train=x_tr, num_tasks=num_tasks, num_latents=NUM_LATENTS,
        num_inducing=NUM_INDUCING,
        kernel_type=KERNEL, kernel_nu=KERNEL_NU,
    ).to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks).to(device)

    print(f"  Model params: {count_params(model):,} | "
          f"Likelihood params: {count_params(likelihood):,}")

    model.train(); likelihood.train()
    mll = VariationalELBO(likelihood, model, num_data=y_tr.size(0))
    mse_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    loader = DataLoader(TensorDataset(x_tr, y_tr),
                        batch_size=BATCH_SIZE, shuffle=True)
    elbo_hist, mse_hist = [], []

    print(f"\nTraining for {NUM_EPOCHS} epochs ...\n")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        te = time.time(); ep_elbo = ep_mse = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = -mll(out, by)
            loss_mse = mse_fn(out.mean, by)
            loss.backward(); optimizer.step()
            ep_elbo += loss.item(); ep_mse += loss_mse.item()
        scheduler.step()
        nb = len(loader)
        elbo_hist.append(ep_elbo / nb); mse_hist.append(ep_mse / nb)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>4d}/{NUM_EPOCHS} | "
                  f"elbo: {elbo_hist[-1]:.4f} | mse: {mse_hist[-1]:.6f} | "
                  f"time: {time.time()-te:.1f}s")
    print(f"\nTraining done in {time.time()-t0:.1f}s")

    model.eval(); likelihood.eval()
    all_m, all_v = [], []
    with torch.no_grad():
        for bx, _ in DataLoader(TensorDataset(x_t, x_t),
                                batch_size=BATCH_SIZE, shuffle=False):
            out = model(bx)
            m, v = out.mean, out.variance
            if m.ndim == 3 and m.shape[1] == 1:
                m, v = m.squeeze(1), v.squeeze(1)
            all_m.append(m.cpu()); all_v.append(v.cpu())
    mean_pred = torch.cat(all_m, 0); var_pred = torch.cat(all_v, 0)

    y_t_cpu = y_t.cpu()
    mse = torch.nn.functional.mse_loss(mean_pred, y_t_cpu).item()
    rel = relative_l2_error(mean_pred, y_t_cpu).item()
    print(f"\n  MSE: {mse:.6f} | Relative L2: {100*rel:.2f}%")

    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(SAVE_DIR, f"svgp_{ts}.npz"),
             mean_pred=mean_pred.numpy(), var_pred=var_pred.numpy(),
             y_test=y_t_cpu.numpy(),
             elbo_history=np.array(elbo_hist), mse_history=np.array(mse_hist))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"svgp_{ts}.pt"))
    print(f"  Saved to {SAVE_DIR}/svgp_{ts}.*")


if __name__ == "__main__":
    main()
