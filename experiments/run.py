"""Universal experiment runner for SGPO.

Usage:
    python experiments/run.py --config configs/burger.yaml
    python experiments/run.py --config configs/darcy.yaml --device cuda:0
    python experiments/run.py --config configs/burger.yaml --optimizer ngd
"""
import sys
import os
import argparse

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gpytorch

from sgpo.config.loader import load_config
from sgpo.utils.misc import set_seed, resolve_device, count_params
from sgpo.data.loaders import load_data
from sgpo.models.vnngp import VNNGP
from sgpo.models.svgp import SVGP
from sgpo.wno.wno_models import WNO1d, WNO2d
from sgpo.wno.mean import WNOMean
from sgpo.wno.embedding import WNOEmbedding
from sgpo.kernels.registry import build_kernel
from sgpo.means.nn_mean import NNMean
from sgpo.means.basis_mean import FourierBasisMean, PolynomialBasisMean
from sgpo.training.trainer import Trainer


def build_wno(cfg, device):
    """Build a WNO model based on config. Returns None if not needed."""
    mcfg = cfg.model
    dcfg = cfg.data
    if mcfg.mean_type != "wno" and not mcfg.use_wno_embedding:
        return None

    res = dcfg.resolution
    if len(res) == 1:
        wno = WNO1d(
            width=mcfg.wno_width,
            level=mcfg.wno_level,
            layers=mcfg.wno_layers,
            size=res[0],
            wavelet=mcfg.wno_wavelet,
        )
    else:
        wno = WNO2d(
            width=mcfg.wno_width,
            level=mcfg.wno_level,
            layers=mcfg.wno_layers,
            size=res,
            wavelet=mcfg.wno_wavelet,
        )
    return wno.to(device)


def build_model(cfg, x_train, device):
    """Build GP model + likelihood from config.

    Returns (model, likelihood) on the specified device.
    """
    mcfg = cfg.model
    dcfg = cfg.data

    # Number of output dimensions — infer from the actual data shape
    # instead of relying solely on config.resolution, which may be stale.
    num_tasks = x_train.shape[-1]

    # Optional WNO components
    wno = build_wno(cfg, device)
    mean_module = None
    wno_embedding = None

    if wno is not None:
        if mcfg.mean_type == "wno":
            mean_module = WNOMean(wno)
        if mcfg.use_wno_embedding:
            embed_dim = getattr(mcfg, 'wno_embed_dim', 32)
            wno_embedding = WNOEmbedding(wno, embed_dim=embed_dim)

    # Non-WNO mean types
    if mcfg.mean_type == "nn":
        mean_module = NNMean(
            input_dim=num_tasks,
            output_dim=num_tasks,
            hidden_dim=getattr(mcfg, "nn_mean_hidden", 128),
            num_hidden=getattr(mcfg, "nn_mean_layers", 3),
        )
    elif mcfg.mean_type == "fourier_basis":
        mean_module = FourierBasisMean(
            num_modes=getattr(mcfg, "fourier_modes", 8),
        )
    elif mcfg.mean_type == "polynomial_basis":
        mean_module = PolynomialBasisMean(
            input_dim=num_tasks,
            degree=getattr(mcfg, "poly_degree", 3),
        )

    # Build GP model
    if mcfg.model_type == "vnngp":
        model = VNNGP(
            x_train=x_train,
            num_tasks=num_tasks,
            num_latents=mcfg.num_latents,
            num_nn=mcfg.num_nn,
            num_inducing=mcfg.num_inducing,
            training_batch_size=cfg.training.batch_size,
            kernel_type=mcfg.kernel,
            kernel_nu=mcfg.kernel_nu,
            use_ard=mcfg.use_ard,
            mean_module=mean_module,
            wno_embedding=wno_embedding,
        )
    elif mcfg.model_type == "svgp":
        model = SVGP(
            x_train=x_train,
            num_tasks=num_tasks,
            num_latents=mcfg.num_latents,
            num_inducing=mcfg.num_inducing,
            kernel_type=mcfg.kernel,
            kernel_nu=mcfg.kernel_nu,
            use_ard=mcfg.use_ard,
            mean_module=mean_module,
            wno_embedding=wno_embedding,
        )
    else:
        raise ValueError(f"Unknown model_type: {mcfg.model_type}")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = model.to(device)
    likelihood = likelihood.to(device)

    return model, likelihood


def main():
    parser = argparse.ArgumentParser(description="SGPO experiment runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. cuda:0)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="Override optimizer (adamw|adam|ngd|lbfgs|muon|schedule_free)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.optimizer is not None:
        cfg.training.optimizer = args.optimizer
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs

    # Setup
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.device)

    # Load data
    x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer = load_data(cfg.data, device)

    # Build model
    model, likelihood = build_model(cfg, x_tr, device)
    n_params = count_params(model)
    n_params_lik = count_params(likelihood)

    # Train
    trainer = Trainer(model, likelihood, cfg, x_normalizer, y_normalizer)
    trainer.logger.log_header(
        device=device,
        n_params=n_params,
        n_params_lik=n_params_lik,
        data_shapes=f"x_tr={list(x_tr.shape)}, y_tr={list(y_tr.shape)}, "
                    f"x_t={list(x_t.shape)}, y_t={list(y_t.shape)}",
    )
    trainer.train(x_tr, y_tr)

    # Predict + evaluate
    mean_pred, var_pred = trainer.predict(x_t)
    metrics = trainer.evaluate(mean_pred, y_t)

    # Save
    trainer.save(mean_pred, var_pred, y_t)

    print(f"\nDone. MSE={metrics['mse']:.6f} | Relative L2={metrics['relative_l2_error_pct']:.2f}%")


if __name__ == "__main__":
    main()
