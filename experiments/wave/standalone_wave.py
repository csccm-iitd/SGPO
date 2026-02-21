#!/usr/bin/env python
"""Standalone wave equation experiment.

Runs the full VNNGP pipeline for 1-D wave equation:
    u(x, t=0) -> u(x, t=80)

All hyperparameters are defined as constants at the top of this file.
No YAML config needed -- just run:
    python experiments/wave/standalone_wave.py
"""

import sys
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import gpytorch
from gpytorch.mlls import VariationalELBO

# Path setup so sgpo package is importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sgpo.models.vnngp import VNNGP
from sgpo.wno.wno_models import WNO1d
from sgpo.wno.embedding import WNOEmbedding
from sgpo.utils.metrics import relative_l2_error
from sgpo.utils.misc import set_seed, resolve_device, count_params

# Hyperparameters (mirroring configs/wave.yaml)

# --- Data ---
TRAIN_PATH = "/DATA/Sawan_projects/research/DATA/train_IC2.npz"
TEST_PATH = "/DATA/Sawan_projects/research/DATA/test_IC2.npz"
NTRAIN = 1000
NTEST = 100
T_IN = 0
T_OUT = 80
RESOLUTION = [200]

# --- Model ---
NUM_LATENTS = 60
NUM_NN = 70
NUM_INDUCING = 1000
KERNEL = "matern"
KERNEL_NU = 2.5
USE_ARD = True

# --- WNO embedding ---
WNO_WIDTH = 64
WNO_LEVEL = 4
WNO_LAYERS = 4
WNO_WAVELET = "db4"

# --- Training ---
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 200
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.75

# --- Misc ---
SEED = 0
DEVICE = "cuda:1"
SAVE_DIR = "results/wave"


# Main

def main():
    # 1. Setup
    set_seed(SEED)
    device = resolve_device(DEVICE)

    print("=" * 70)
    print("  Standalone Wave Equation Experiment")
    print(f"  Device     : {device}")
    print(f"  Resolution : {RESOLUTION}")
    print(f"  t_in={T_IN}, t_out={T_OUT}")
    print(f"  Train/Test : {NTRAIN} / {NTEST}")
    print(f"  Epochs     : {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  LR         : {LR} | WD: {WEIGHT_DECAY}")
    print(f"  Kernel     : {KERNEL} (nu={KERNEL_NU})")
    print(f"  Latents    : {NUM_LATENTS} | NN: {NUM_NN} | Inducing: {NUM_INDUCING}")
    print("=" * 70)

    # 2. Load data from .npz
    data_train = np.load(TRAIN_PATH)
    data_test = np.load(TEST_PATH)

    u_train = data_train["u"]  # (n_samples, n_time, n_space)
    u_test = data_test["u"]

    # Input: IC at t=T_IN; Output: solution at t=T_OUT
    x_train = torch.tensor(u_train[:NTRAIN, T_IN, :]).float()
    y_train = torch.tensor(u_train[:NTRAIN, T_OUT, :]).float()
    x_test = torch.tensor(u_test[:NTEST, T_IN, :]).float()
    y_test = torch.tensor(u_test[:NTEST, T_OUT, :]).float()

    print(f"\n  x_train: {list(x_train.shape)}, y_train: {list(y_train.shape)}")
    print(f"  x_test:  {list(x_test.shape)},  y_test:  {list(y_test.shape)}")

    # No normalization for Wave
    # Flatten for GP (already 1D)
    x_tr = x_train.reshape(NTRAIN, -1).to(device)
    y_tr = y_train.reshape(NTRAIN, -1).to(device)
    x_t = x_test.reshape(NTEST, -1).to(device)
    y_t = y_test.reshape(NTEST, -1).to(device)

    # 3. Build model + likelihood
    num_tasks = x_tr.shape[-1]

    # Build WNO embedding for kernel: K(Phi(x1), Phi(x2))
    wno = WNO1d(
        width=WNO_WIDTH, level=WNO_LEVEL, layers=WNO_LAYERS,
        size=RESOLUTION[0], wavelet=WNO_WAVELET,
    ).to(device)
    wno_embed = WNOEmbedding(wno)

    model = VNNGP(
        x_train=x_tr,
        num_tasks=num_tasks,
        num_latents=NUM_LATENTS,
        num_nn=NUM_NN,
        num_inducing=NUM_INDUCING,
        training_batch_size=BATCH_SIZE,
        kernel_type=KERNEL,
        kernel_nu=KERNEL_NU,
        use_ard=USE_ARD,
        wno_embedding=wno_embed,
    ).to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
    ).to(device)

    n_params = count_params(model)
    n_params_lik = count_params(likelihood)
    print(f"\n  Model params    : {n_params:,}")
    print(f"  Likelihood params: {n_params_lik:,}")

    # 4. Training loop
    model.train()
    likelihood.train()

    mll = VariationalELBO(likelihood, model, num_data=y_tr.size(0))
    mse_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA,
    )

    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    elbo_history = []
    mse_history = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs ...\n")
    train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_elbo = 0.0
        epoch_mse = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y)
            loss_mse = mse_fn(output.mean, batch_y)
            loss.backward()
            optimizer.step()

            epoch_elbo += loss.item()
            epoch_mse += loss_mse.item()

        scheduler.step()

        n_batches = len(train_loader)
        avg_elbo = epoch_elbo / n_batches
        avg_mse = epoch_mse / n_batches
        elbo_history.append(avg_elbo)
        mse_history.append(avg_mse)
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:>4d}/{NUM_EPOCHS} | "
                f"elbo: {avg_elbo:.4f} | mse: {avg_mse:.6f} | "
                f"time: {epoch_time:.1f}s"
            )

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time / 60:.1f}min)")

    # 5. Prediction
    model.eval()
    likelihood.eval()

    test_loader = DataLoader(
        TensorDataset(x_t, x_t), batch_size=BATCH_SIZE, shuffle=False,
    )

    all_means = []
    all_vars = []

    print("\nRunning predictions ...")
    with torch.no_grad():
        for batch_x, _ in test_loader:
            output = model(batch_x)
            mean_batch = output.mean
            var_batch = output.variance

            if mean_batch.ndim == 3 and mean_batch.shape[1] == 1:
                mean_batch = mean_batch.squeeze(1)
                var_batch = var_batch.squeeze(1)

            all_means.append(mean_batch.cpu())
            all_vars.append(var_batch.cpu())

    mean_pred = torch.cat(all_means, dim=0)
    var_pred = torch.cat(all_vars, dim=0)

    # No normalization to decode for Wave

    # 6. Evaluate
    y_t_cpu = y_t.detach().cpu()
    mse = torch.nn.functional.mse_loss(mean_pred, y_t_cpu).item()
    rel_err = relative_l2_error(mean_pred, y_t_cpu).item()

    print("-" * 50)
    print("  Evaluation Results:")
    print(f"    MSE              : {mse:.6f}")
    print(f"    Relative L2 error: {rel_err:.6f}")
    print(f"    Relative L2 (%)  : {100.0 * rel_err:.2f}%")
    print("-" * 50)

    # 7. Save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save predictions + losses as .npz
    npz_path = os.path.join(SAVE_DIR, f"results_{timestamp}.npz")
    np.savez(
        npz_path,
        mean_pred=mean_pred.detach().cpu().numpy(),
        var_pred=var_pred.detach().cpu().numpy(),
        y_test=y_t_cpu.numpy(),
        elbo_history=np.array(elbo_history),
        mse_history=np.array(mse_history),
    )
    print(f"\nResults saved to {npz_path}")

    # Save model checkpoint as .pt
    pt_path = os.path.join(SAVE_DIR, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Model saved to {pt_path}")

    print(f"\nDone. MSE={mse:.6f} | Relative L2={100.0 * rel_err:.2f}%")


if __name__ == "__main__":
    main()
