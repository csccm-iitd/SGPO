#!/usr/bin/env python
"""Standalone Burgers experiment — Exact GP + WNO mean (no kernel embedding).

Full-batch exact GP. The WNO learns the mean, and a shared Matern kernel
models residuals with calibrated uncertainty.

    a(x) -> u(x, t=1)

Usage:
    python experiments/burger/standalone_exactgp_wno_burger.py
"""

import sys, os, time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sgpo.models.exact_gp import ExactGPModel
from sgpo.wno.wno_models import WNO1d
from sgpo.data.readers import MatReader
from sgpo.utils.metrics import relative_l2_error
from sgpo.utils.misc import set_seed, resolve_device, count_params

# ── Hyperparameters ──────────────────────────────────────────────────

DATA_PATH = "/DATA/Sawan_projects/research/DATA/burgers_data_R10.mat"
X_FIELD, Y_FIELD = "a", "u"
NTRAIN, NTEST, SUB = 1000, 100, 8

# WNO mean
WNO_WIDTH = 32
WNO_LEVEL = 4
WNO_LAYERS = 4
WNO_WAVELET = "db4"

# GP kernel
KERNEL_TYPE = "matern"
KERNEL_NU = 2.5
NOISE_INIT = 0.1

# Training
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.75

SEED = 0
DEVICE = "cuda:1"
SAVE_DIR = "results/burger"


def main():
    set_seed(SEED)
    device = resolve_device(DEVICE)

    print("=" * 70)
    print("  Burgers — Exact GP + WNO mean")
    print(f"  Device: {device} | Epochs: {NUM_EPOCHS}")
    print(f"  WNO width={WNO_WIDTH}, layers={WNO_LAYERS}")
    print("=" * 70)

    # ── Data ──
    reader = MatReader(DATA_PATH)
    x_data = reader.read_field(X_FIELD)[:, ::SUB]
    y_data = reader.read_field(Y_FIELD)[:, ::SUB]
    x_train, y_train = x_data[:NTRAIN], y_data[:NTRAIN]
    x_test, y_test = x_data[-NTEST:], y_data[-NTEST:]
    size = x_train.shape[1]
    print(f"  x_train: {list(x_train.shape)}, y_train: {list(y_train.shape)}")

    x_tr = x_train.reshape(NTRAIN, -1).to(device)
    y_tr = y_train.reshape(NTRAIN, -1).to(device)
    x_t = x_test.reshape(NTEST, -1).to(device)
    y_t = y_test.reshape(NTEST, -1).to(device)

    # ── Model ──
    wno = WNO1d(width=WNO_WIDTH, level=WNO_LEVEL, layers=WNO_LAYERS,
                size=size, wavelet=WNO_WAVELET).to(device)

    model = ExactGPModel(
        mean_fn=wno, kernel_type=KERNEL_TYPE,
        kernel_nu=KERNEL_NU, noise_init=NOISE_INIT,
    ).to(device)

    print(f"  Model params: {count_params(model):,}")

    # ── Training (full-batch) ──
    model.train()
    mse_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    mll_hist, mse_hist = [], []

    print(f"\nTraining for {NUM_EPOCHS} epochs ...\n")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        te = time.time()
        optimizer.zero_grad()
        neg_mll = -model.compute_mll(x_tr, y_tr)
        neg_mll.backward()
        optimizer.step()
        scheduler.step()

        mll_hist.append(neg_mll.item())

        # MSE tracking
        with torch.no_grad():
            mean_tr = model._compute_mean(x_tr)
            train_mse = mse_fn(mean_tr, y_tr).item()
        mse_hist.append(train_mse)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>4d}/{NUM_EPOCHS} | "
                  f"neg_mll: {mll_hist[-1]:.4f} | mse: {train_mse:.6f} | "
                  f"noise: {model.noise.item():.4f} | "
                  f"time: {time.time()-te:.1f}s")

    print(f"\nTraining done in {time.time()-t0:.1f}s")

    # ── Prediction ──
    model.eval()
    pred_mean, pred_var = model.predict(x_tr, y_tr, x_t)
    pred_mean = pred_mean.cpu()
    pred_var = pred_var.cpu()

    y_t_cpu = y_t.cpu()
    mse = torch.nn.functional.mse_loss(pred_mean, y_t_cpu).item()
    rel = relative_l2_error(pred_mean, y_t_cpu).item()
    print(f"\n  MSE: {mse:.6f} | Relative L2: {100*rel:.2f}%")

    # ── Save ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(SAVE_DIR, f"exactgp_wno_{ts}.npz"),
             mean_pred=pred_mean.numpy(), var_pred=pred_var.numpy(),
             y_test=y_t_cpu.numpy(),
             mll_history=np.array(mll_hist), mse_history=np.array(mse_hist))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"exactgp_wno_{ts}.pt"))
    print(f"  Saved to {SAVE_DIR}/exactgp_wno_{ts}.*")


if __name__ == "__main__":
    main()
