"""Enhanced WNO with multi-level wavelet-Fourier layers; same interface as FNO."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sgpo.operators.fno import FNO
from sgpo.operators.wavelet_fourier import WaveConv1dML, WaveConv2dML


class EWNO(nn.Module):
    def __init__(self, grid, width=32, level=3, layers=4, wavelet="db4", modes=12, in_ch=1, out_ch=1, pad=0, proj=128):
        super().__init__()
        self.dim, self.pad = len(grid), pad
        size = [g + pad for g in grid]
        conv = WaveConv1dML if self.dim == 1 else WaveConv2dML
        lin = nn.Conv1d if self.dim == 1 else nn.Conv2d
        self.lift = nn.Linear(in_ch + self.dim, width)
        self.wave = nn.ModuleList(conv(width, level, size[0] if self.dim == 1 else size, wavelet, modes) for _ in range(layers))
        self.skip = nn.ModuleList(lin(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, out_ch))

    def features(self, x):
        h = self.lift(torch.cat([x, FNO.grid(x)], -1)).movedim(-1, 1)
        if self.pad:
            h = F.pad(h, [0, self.pad] * self.dim)
        for i, (w, s) in enumerate(zip(self.wave, self.skip)):
            h = w(h)[..., *[slice(0, n) for n in h.shape[2:]]] + s(h)
            if i < len(self.wave) - 1:
                h = F.gelu(h)
        if self.pad:
            h = h[..., :-self.pad] if self.dim == 1 else h[..., :-self.pad, :-self.pad]
        return h.movedim(1, -1)

    def forward(self, x):
        return self.proj(self.features(x))
