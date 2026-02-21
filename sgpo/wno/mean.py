import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import gpytorch


class WNOMean(gpytorch.means.Mean):
    """WNO-based non-zero mean function for GP.

    Uses gradient checkpointing on chunks so that intermediate wavelet
    activations are freed after each chunk's forward pass and recomputed
    during backward, keeping peak GPU memory tractable.
    """

    MAX_CHUNK = 256

    def __init__(self, wno: nn.Module):
        super().__init__()
        self.wno = wno

    def _chunk_forward(self, x_chunk: torch.Tensor) -> torch.Tensor:
        return self.wno(x_chunk)

    def forward(self, x):
        leading_shape = x.shape[:-1]
        d = x.shape[-1]
        flat = x.reshape(-1, d)
        n = flat.shape[0]

        if n <= self.MAX_CHUNK:
            out = self._chunk_forward(flat)
        else:
            parts = []
            for i in range(0, n, self.MAX_CHUNK):
                chunk = flat[i : i + self.MAX_CHUNK]
                if self.training:
                    parts.append(
                        cp.checkpoint(
                            self._chunk_forward, chunk, use_reentrant=False
                        )
                    )
                else:
                    parts.append(self._chunk_forward(chunk))
            out = torch.cat(parts, dim=0)

        out_dim = out.shape[-1]
        return out.reshape(*leading_shape, out_dim)
