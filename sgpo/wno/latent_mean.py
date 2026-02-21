

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import gpytorch


class LatentWNOMean(gpytorch.means.Mean):
    """Scalar WNO-feature mean for latent GPs in LMC framework."""

    MAX_CHUNK = 256

    def __init__(self, wno: nn.Module):
        super().__init__()
        self.wno = wno
        self.width = wno.width

        if isinstance(wno.size, (list, tuple)):
            self.spatial_shape = tuple(wno.size)
        else:
            self.spatial_shape = (wno.size,)

        self.proj = nn.Linear(self.width, 1, bias=True)

    # ----- chunked WNO → pool → scalar -----

    def _chunk_fn(self, x_chunk: torch.Tensor) -> torch.Tensor:
        feat = self.wno.get_features(x_chunk)           # (chunk, width * spatial)
        feat = feat.reshape(-1, self.width, *self.spatial_shape)
        pool_dims = tuple(range(2, 2 + len(self.spatial_shape)))
        feat = feat.mean(dim=pool_dims)                  # (chunk, width)
        return self.proj(feat).squeeze(-1)               # (chunk,)

    # ----- forward -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar mean per sample, preserving leading batch dims.

        Parameters
        ----------
        x : Tensor  ``(*leading, input_dim)``
            E.g. ``(num_latents, batch, d)`` from LMC, or ``(batch, d)``.

        Returns
        -------
        Tensor  ``(*leading,)``
            Scalar mean per sample per (optional) latent dimension.
        """
        leading = x.shape[:-1]
        d = x.shape[-1]
        flat = x.reshape(-1, d)
        n = flat.shape[0]

        if n <= self.MAX_CHUNK:
            out = self._chunk_fn(flat)
        else:
            parts = []
            for i in range(0, n, self.MAX_CHUNK):
                chunk = flat[i : i + self.MAX_CHUNK]
                if self.training:
                    parts.append(
                        cp.checkpoint(
                            self._chunk_fn, chunk, use_reentrant=False
                        )
                    )
                else:
                    parts.append(self._chunk_fn(chunk))
            out = torch.cat(parts, dim=0)

        return out.reshape(leading)
