"""Exact GP with shared kernel and learned mean function (e.g. WNO).

The WNO mean captures the nonlinear PDE operator mapping.
The GP models residuals and provides calibrated uncertainty.
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class ExactGPModel(nn.Module):
    """Exact GP with shared Matern/RBF kernel and learned mean function.

    Parameters
    ----------
    mean_fn : nn.Module
        Callable (n, d) → (n, d_out), e.g. ``WNO1d`` or ``WNO2d``.
    kernel_type : str
        'matern' or 'rbf'.
    kernel_nu : float
        Matern smoothness (0.5, 1.5, 2.5).
    noise_init : float
        Initial observation noise variance.
    """

    MEAN_CHUNK = 256

    def __init__(self, mean_fn, kernel_type="matern", kernel_nu=2.5,
                 noise_init=0.1):
        super().__init__()
        self.mean_fn = mean_fn
        self.kernel_type = kernel_type
        self.kernel_nu = kernel_nu

        self.log_noise = nn.Parameter(torch.tensor(math.log(noise_init)))
        self.log_outputscale = nn.Parameter(torch.tensor(0.0))
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))

    # Properties
    @property
    def noise(self):
        return self.log_noise.exp()

    @property
    def outputscale(self):
        return self.log_outputscale.exp()

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    # Kernel
    def kernel_matrix(self, x1, x2):
        """Scaled Matern / RBF kernel matrix (n1, n2)."""
        x1_s = x1 / self.lengthscale
        x2_s = x2 / self.lengthscale
        dist = torch.cdist(x1_s, x2_s, p=2)

        if self.kernel_type == "matern" and self.kernel_nu == 2.5:
            s = math.sqrt(5) * dist
            K = (1.0 + s + s.pow(2) / 3.0) * (-s).exp()
        elif self.kernel_type == "matern" and self.kernel_nu == 1.5:
            s = math.sqrt(3) * dist
            K = (1.0 + s) * (-s).exp()
        elif self.kernel_type == "matern" and self.kernel_nu == 0.5:
            K = (-dist).exp()
        else:  # RBF
            K = (-0.5 * dist.pow(2)).exp()

        return self.outputscale * K

    # Mean with chunking
    def _compute_mean(self, x):
        """WNO forward with memory-saving chunking."""
        n = x.shape[0]
        if n <= self.MEAN_CHUNK:
            return self.mean_fn(x)
        parts = []
        for i in range(0, n, self.MEAN_CHUNK):
            chunk = x[i:i + self.MEAN_CHUNK]
            if self.training:
                parts.append(
                    cp.checkpoint(self.mean_fn, chunk, use_reentrant=False)
                )
            else:
                parts.append(self.mean_fn(chunk))
        return torch.cat(parts, dim=0)

    # Training
    def compute_mll(self, x_train, y_train):
        """Exact marginal log-likelihood (scalar, averaged per datum)."""
        n, d_out = y_train.shape

        mean = self._compute_mean(x_train)       # (n, d_out)
        residuals = y_train - mean

        K = self.kernel_matrix(x_train, x_train)  # (n, n)
        K_noise = K + self.noise * torch.eye(n, device=x_train.device)

        L = torch.linalg.cholesky(K_noise)
        alpha = torch.cholesky_solve(residuals, L)  # (n, d_out)

        data_fit = -0.5 * (residuals * alpha).sum()
        complexity = -d_out * L.diagonal().log().sum()
        constant = -0.5 * n * d_out * math.log(2 * math.pi)

        return (data_fit + complexity + constant) / (n * d_out)

    # Prediction
    @torch.no_grad()
    def predict(self, x_train, y_train, x_test):
        """Posterior predictive mean and variance on *x_test*.

        Returns
        -------
        pred_mean : (n_test, d_out)
        pred_var  : (n_test, d_out)  — diagonal variance (same per output).
        """
        n = x_train.shape[0]

        mean_train = self._compute_mean(x_train)
        mean_test = self._compute_mean(x_test)

        residuals = y_train - mean_train

        K = self.kernel_matrix(x_train, x_train)
        K += self.noise * torch.eye(n, device=x_train.device)
        K_star = self.kernel_matrix(x_test, x_train)  # (n_test, n)

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(residuals, L)     # (n, d_out)

        pred_mean = mean_test + K_star @ alpha          # (n_test, d_out)

        # Predictive variance (shared across outputs because kernel is shared)
        v = torch.linalg.solve_triangular(L, K_star.T, upper=False)  # (n, n_test)
        pred_var_diag = self.outputscale - (v * v).sum(dim=0)         # (n_test,)
        pred_var_diag = pred_var_diag.clamp_min(1e-6)

        return pred_mean, pred_var_diag.unsqueeze(1).expand_as(pred_mean)
