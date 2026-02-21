"""Kernel wrapper 
"""

import torch
from gpytorch.kernels import Kernel


class EmbeddingKernel(Kernel):
    """Kernel that first embeds inputs through a feature extractor.

    K(x1, x2) = base_kernel(embedding(x1), embedding(x2)).

    Parameters
    ----------
    base_kernel : gpytorch.kernels.Kernel
        Any GPyTorch kernel (e.g. ScaleKernel(MaternKernel(...))).
    embedding : torch.nn.Module
        Callable mapping ``(... , D_in) -> (... , D_out)`` (e.g. WNOEmbedding).
    """

    def __init__(self, base_kernel: Kernel, embedding: torch.nn.Module):
        # has_lengthscale=False because the *base* kernel already owns any
        # lengthscale / outputscale parameters.
        super().__init__(has_lengthscale=False)
        self.base_kernel = base_kernel
        self.embedding = embedding

    # ------------------------------------------------------------------
    # GPyTorch's Kernel.dtype iterates over *all* parameters.  WNO uses
    # pytorch_wavelets which stores complex64 filter coefficients.  This
    # confuses GPyTorch which wants a single dtype.  We override to
    # report only the base kernel's dtype.
    # ------------------------------------------------------------------
    @property
    def dtype(self):
        return self.base_kernel.dtype

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        z1 = self.embedding(x1)
        # Avoid redundant embedding when x1 is x2 (same object)
        z2 = z1 if x1 is x2 else self.embedding(x2)
        return self.base_kernel.forward(
            z1, z2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
