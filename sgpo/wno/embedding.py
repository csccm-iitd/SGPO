import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class WNOEmbedding(nn.Module):


    MAX_CHUNK = 256

    def __init__(self, wno: nn.Module, embed_dim: int = 32):
        super().__init__()
        self.wno = wno
        self.width = wno.width

        if isinstance(wno.size, (list, tuple)):
            self.spatial_shape = tuple(wno.size)
        else:
            self.spatial_shape = (wno.size,)

        if embed_dim > 0 and embed_dim != self.width:
            self.proj = nn.Linear(self.width, embed_dim, bias=False)
            self.embed_dim = embed_dim
        else:
            self.proj = None
            self.embed_dim = self.width

    def _embed_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        feat = self.wno.get_features(x_chunk)          # (chunk, width * spatial)
        feat = feat.reshape(-1, self.width, *self.spatial_shape)
        pool_dims = tuple(range(2, 2 + len(self.spatial_shape)))
        feat = feat.mean(dim=pool_dims)                 # (chunk, width)
        if self.proj is not None:
            feat = self.proj(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        d = x.shape[-1]
        flat = x.reshape(-1, d)
        n = flat.shape[0]

        if n <= self.MAX_CHUNK:
            out = self._embed_chunk(flat)
        else:
            parts = []
            for i in range(0, n, self.MAX_CHUNK):
                chunk = flat[i : i + self.MAX_CHUNK]
                if self.training:
                    parts.append(
                        cp.checkpoint(
                            self._embed_chunk, chunk, use_reentrant=False
                        )
                    )
                else:
                    parts.append(self._embed_chunk(chunk))
            out = torch.cat(parts, dim=0)

        return out.reshape(*leading_shape, self.embed_dim)
