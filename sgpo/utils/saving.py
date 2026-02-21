import os
import json
import numpy as np
import torch
from datetime import datetime
from dataclasses import asdict


def save_results(save_dir, config, model=None, mean_pred=None, var_pred=None,
                 y_test=None, elbo_history=None, mse_history=None):
    """Save experiment results: .npz for data, .pt for model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Save predictions + losses + metadata as .npz ---
    npz_dict = {}
    if mean_pred is not None:
        npz_dict["mean_pred"] = mean_pred.detach().cpu().numpy()
    if var_pred is not None:
        npz_dict["var_pred"] = var_pred.detach().cpu().numpy()
    if y_test is not None:
        npz_dict["y_test"] = y_test.detach().cpu().numpy()
    if elbo_history is not None:
        npz_dict["elbo_history"] = np.array(elbo_history)
    if mse_history is not None:
        npz_dict["mse_history"] = np.array(mse_history)
    # Store config as a JSON string
    npz_dict["config_json"] = np.array(json.dumps(asdict(config), indent=2))

    npz_path = os.path.join(save_dir, f"results_{timestamp}.npz")
    np.savez(npz_path, **npz_dict)
    print(f"Results saved to {npz_path}")

    # --- Save model state dict as .pt ---
    if model is not None and config.logging.save_model:
        pt_path = os.path.join(save_dir, f"model_{timestamp}.pt")
        torch.save(model.state_dict(), pt_path)
        print(f"Model saved to {pt_path}")

    return save_dir
