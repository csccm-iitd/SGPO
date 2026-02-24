#!/usr/bin/env python
"""Standalone Darcy triangular-notch experiment.

Runs the full VNNGP pipeline for 2-D Darcy flow with triangular notch:
    boundCoeff(x,y) -> sol(x,y)

All hyperparameters are defined as constants at the top of this file.
No YAML config or Trainer abstraction is used -- the entire pipeline
is inlined in main().

Usage (from project root or this directory):
    python experiments/darcy_notch/standalone_darcy_notch.py
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
from sgpo.data.readers import MatReader
from sgpo.data.normalizers import UnitGaussianNormalizer
from sgpo.utils.metrics import relative_l2_error
from sgpo.utils.misc import set_seed, resolve_device, count_params

# Hyperparameters (mirroring configs/darcy_notch.yaml)

# --- Data ---
DATA_PATH = "/DATA/Sawan_projects/research/DATA/Darcy_Triangular_FNO.mat"
X_FIELD = "boundCoeff"
Y_FIELD = "sol"
NTRAIN = 1500
NTEST = 100
SUB = 3
RESOLUTION = [34, 34]
NORMALIZE_X = True
NORMALIZE_Y = True

# --- Model ---
NUM_LATENTS = 120
NUM_NN = 90
NUM_INDUCING = 500
KERNEL = "matern"
KERNEL_NU = 2.5

# --- Training ---
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 500
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.75

# --- Misc ---
SEED = 0
DEVICE = "auto"
SAVE_DIR = "results/darcy_notch"


# Main

def main():
    # 1. Setup
    set_seed(SEED)
    device = resolve_device(DEVICE)
    print(f"Device: {device}")

    # 2. Load and preprocess data
    r = SUB
    s = RESOLUTION[0]  # 85

    reader = MatReader(DATA_PATH)

    # Train split: first NTRAIN samples
    x_train = reader.read_field(X_FIELD)[:NTRAIN, ::r, ::r][:, :s, :s]
    y_train = reader.read_field(Y_FIELD)[:NTRAIN, ::r, ::r][:, :s, :s]

    # Test split: last NTEST samples
    x_test = reader.read_field(X_FIELD)[-NTEST:, ::r, ::r][:, :s, :s]
    y_test = reader.read_field(Y_FIELD)[-NTEST:, ::r, ::r][:, :s, :s]

    print(f"Raw shapes  -- x_train: {list(x_train.shape)}, y_train: {list(y_train.shape)}")
    print(f"               x_test:  {list(x_test.shape)},  y_test:  {list(y_test.shape)}")

    # Normalize BEFORE flattening (pointwise statistics over the grid)
    x_normalizer, y_normalizer = None, None
    if NORMALIZE_X:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if NORMALIZE_Y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    # Flatten to (n, s*s) for GP
    x_tr = x_train.reshape(NTRAIN, -1).to(device)
    y_tr = y_train.reshape(NTRAIN, -1).to(device)
    x_t = x_test.reshape(NTEST, -1).to(device)
    y_t = y_test.reshape(NTEST, -1).to(device)

    print(f"Flat shapes -- x_tr: {list(x_tr.shape)}, y_tr: {list(y_tr.shape)}")
    print(f"               x_t:  {list(x_t.shape)},  y_t:  {list(y_t.shape)}")

    # Move normalizers to device for later decoding
    if y_normalizer is not None:
        y_normalizer.to(device)

    # 3. Build model + likelihood
    num_tasks = x_tr.shape[-1]  # s * s

    model = VNNGP(
        x_train=x_tr,
        num_tasks=num_tasks,
        num_latents=NUM_LATENTS,
        num_nn=NUM_NN,
        num_inducing=NUM_INDUCING,
        training_batch_size=BATCH_SIZE,
        kernel_type=KERNEL,
        kernel_nu=KERNEL_NU,
        use_ard=False,
        mean_module=None,
        wno_embedding=None,
    )

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    model = model.to(device)
    likelihood = likelihood.to(device)

    n_params = count_params(model)
    n_params_lik = count_params(likelihood)
    print(f"Model params: {n_params:,} | Likelihood params: {n_params_lik:,}")

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
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA
    )

    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    elbo_history = []
    mse_history = []

    print(f"\n{'='*60}")
    print(f"Training for {NUM_EPOCHS} epochs  (batch_size={BATCH_SIZE})")
    print(f"{'='*60}")

    t_start = time.time()

    for epoch in range(NUM_EPOCHS):
        t_epoch = time.time()
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_epoch
            print(
                f"Epoch {epoch+1:>4d}/{NUM_EPOCHS} | "
                f"ELBO {avg_elbo:+.4f} | MSE {avg_mse:.6f} | "
                f"time {elapsed:.2f}s"
            )

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s")

    # 5. Prediction
    model.eval()
    likelihood.eval()

    test_dataset = TensorDataset(x_t, x_t)  # dummy y
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_means = []
    all_vars = []

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

    # 6. Decode predictions (undo y-normalization)
    if y_normalizer is not None:
        res = RESOLUTION
        std = y_normalizer.std + y_normalizer.eps  # on device

        # Reshape to (n, 85, 85) for pointwise decode, then flatten back
        mean_pred = mean_pred.to(device)
        mean_pred = mean_pred.reshape(mean_pred.shape[0], res[0], res[1])
        mean_pred = y_normalizer.decode(mean_pred)
        mean_pred = mean_pred.reshape(mean_pred.shape[0], -1).cpu()

        # Variance: var_y = std^2 * var_normalized
        var_pred = var_pred.to(device)
        var_pred = var_pred.reshape(var_pred.shape[0], res[0], res[1])
        var_pred = var_pred * (std ** 2)
        var_pred = var_pred.reshape(var_pred.shape[0], -1).cpu()

    # 7. Evaluate
    y_t_cpu = y_t.detach().cpu()
    mse_test = torch.nn.functional.mse_loss(mean_pred, y_t_cpu).item()
    rel_l2 = relative_l2_error(mean_pred, y_t_cpu).item()

    print(f"\n{'='*60}")
    print(f"Evaluation")
    print(f"{'='*60}")
    print(f"Test MSE:          {mse_test:.6f}")
    print(f"Relative L2 error: {rel_l2:.6f}  ({100.0 * rel_l2:.2f}%)")

    # 8. Save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # .npz -- predictions, targets, loss history
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

    # .pt -- model state dict
    pt_path = os.path.join(SAVE_DIR, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Model  saved to {pt_path}")

    print(f"\nDone. MSE={mse_test:.6f} | Relative L2={100.0 * rel_l2:.2f}%")


if __name__ == "__main__":
    main()
