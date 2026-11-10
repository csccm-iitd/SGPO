"""Wavelet Neural Operator (Tripura & Chakraborty 2023) for 1D/2D/3D grids, channel-last like FNO.
3D uses the ptwt transforms (7 detail orientations per level), as in the authors' 3D WNO.

Each wavelet layer takes a J-level DWT, mixes channels location-wise on the coarsest approximation and
detail coefficients (the original WNO kernel), mixes the finer detail levels with a shared channel map
(the original zeroes them), and inverts. With `modes` set, a spectral path is added in parallel (WNO+F).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse, DWTForward, DWTInverse
import ptwt
from sgpo.operators.fno import FNO, SpectralConv


class WaveConv(nn.Module):
    KEYS3 = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

    def __init__(self, width, grid, level, wavelet="db4", fine=True):
        super().__init__()
        self.dim, self.fine = len(grid), fine
        if self.dim == 1:
            self.dwt, self.idwt = DWT1DForward(J=level, wave=wavelet, mode="symmetric"), DWT1DInverse(wave=wavelet, mode="symmetric")
        elif self.dim == 2:
            self.dwt, self.idwt = DWTForward(J=level, wave=wavelet, mode="symmetric"), DWTInverse(wave=wavelet, mode="symmetric")
        else:                                                    # ptwt: (cA, coarsest details, ..., finest details)
            def dwt3(x):
                c = ptwt.wavedec3(x, wavelet, mode="symmetric", level=level)
                return c[0], [torch.stack([d[k] for k in self.KEYS3], 2) for d in reversed(c[1:])]   # finest first, like 2D
            def idwt3(c):
                yl, yh = c
                return ptwt.waverec3((yl, *[{k: h[:, :, i] for i, k in enumerate(self.KEYS3)} for h in reversed(yh)]), wavelet)
            self.dwt, self.idwt = dwt3, idwt3
        with torch.no_grad():
            yl, yh = self.dwt(torch.zeros(1, 1, *grid))
        coarse = yl.shape[2:]
        n_orient = {1: 1, 2: 3, 3: 7}[self.dim]
        scale = 1 / width
        self.w_approx = nn.Parameter(scale * torch.randn(width, width, *coarse))
        self.w_detail = nn.Parameter(scale * torch.randn(n_orient, width, width, *coarse))
        if fine:
            self.w_fine = nn.Parameter(scale * torch.randn(level - 1, n_orient, width, width))
        self.eq = {1: "bix,iox->box", 2: "bixy,ioxy->boxy", 3: "bixyz,ioxyz->boxyz"}[self.dim]

    def forward(self, x):
        yl, yh = self.dwt(x)
        out_l = torch.einsum(self.eq, yl, self.w_approx)
        out_h = []
        for j, c in enumerate(yh):
            if j == len(yh) - 1:
                if self.dim == 1:
                    o = torch.einsum(self.eq, c, self.w_detail[0])
                else:
                    o = torch.stack([torch.einsum(self.eq, c[:, :, k], self.w_detail[k]) for k in range(c.shape[2])], 2)
            elif self.fine:
                if self.dim == 1:
                    o = torch.einsum("bix,io->box", c, self.w_fine[j, 0])
                else:
                    o = torch.einsum("bik...,kio->bok...", c, self.w_fine[j])
            else:
                o = torch.zeros_like(c)
            out_h.append(o)
        y = self.idwt((out_l, out_h))
        return y[(...,) + tuple(slice(0, n) for n in x.shape[2:])]


class WNO(nn.Module):
    """Input (b, *grid, in_ch) -> output (b, *grid, out_ch). `grid` is the unpadded input grid."""

    def __init__(self, grid, width=64, level=3, layers=4, wavelet="db4", in_ch=1, out_ch=1, proj=128, pad=0,
                 modes=None, fine=True, wave_dims=None):
        """wave_dims=2 on a 3D grid applies the wavelet path to the first two axes only (the last axis, e.g. time,
        folded into the batch) and leaves cross-axis coupling to the spectral path; much cheaper than a 3D DWT."""
        super().__init__()
        self.pad, self.dim, self.width = pad, len(grid), width
        self.wave_dims = wave_dims or self.dim
        padded = tuple(n + pad for n in grid)
        self.lift = nn.Linear(in_ch + self.dim, width)
        self.wave = nn.ModuleList(WaveConv(width, padded[: self.wave_dims], level, wavelet, fine) for _ in range(layers))
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(layers)) if modes else None
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
        self.skip = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, out_ch))

    def features(self, x):
        h = self.lift(torch.cat([x, FNO.grid(x)], -1)).movedim(-1, 1)
        if self.pad:
            h = F.pad(h, [0, self.pad] * self.dim)
        for i in range(len(self.wave)):
            z = self._wave(i, h) + self.skip[i](h)
            if self.spectral is not None:
                z = z + self.spectral[i](h)
            h = F.gelu(z) if i < len(self.wave) - 1 else z
        if self.pad:
            h = h[(...,) + tuple(slice(0, n - self.pad) for n in h.shape[2:])]
        return h.movedim(1, -1)

    def _wave(self, i, h):
        if self.wave_dims == self.dim:
            return self.wave[i](h)
        b, c, *g = h.shape                                       # fold the trailing axes into the batch
        k = self.wave_dims
        r = len(g) - k                                           # number of folded (trailing) axes
        hf = h.permute(0, *range(2 + k, 2 + len(g)), 1, *range(2, 2 + k)).reshape(-1, c, *g[:k])   # (b * trail, c, lead)
        out = self.wave[i](hf).reshape(b, *g[k:], c, *g[:k])
        return out.permute(0, 1 + r, *range(2 + r, 2 + r + k), *range(1, 1 + r))                  # back to (b, c, lead, trail)

    def forward(self, x):
        return self.proj(self.features(x))
