"""Fourier Neural Operator for 1D/2D/3D regular grids (Li et al. 2021)."""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.dim = len(modes)
        n_corners = 2 ** (self.dim - 1)
        shape = (width, width, *modes)
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / width ** 2) for _ in range(n_corners)
        )

    def forward(self, x):
        dims = tuple(range(-self.dim, 0))
        size = x.shape[-self.dim:]
        ft = torch.fft.rfftn(x, dim=dims)
        out = torch.zeros_like(ft)
        if self.dim == 1:
            m = self.modes[0]
            out[..., :m] = torch.einsum("bix,iox->box", ft[..., :m], self.weights[0])
        else:
            # low and high (negative) frequencies on every axis but the last (rfft keeps only non-negative there)
            ax = "xyz"[: self.dim]; eq = f"bi{ax},io{ax}->bo{ax}"
            for w, signs in zip(self.weights, itertools.product((0, 1), repeat=self.dim - 1)):
                idx = tuple(slice(-m, None) if s else slice(0, m) for s, m in zip(signs, self.modes[:-1])) + (slice(0, self.modes[-1]),)
                out[(..., *idx)] = torch.einsum(eq, ft[(..., *idx)], w)
        return torch.fft.irfftn(out, s=size, dim=dims)


class FNO(nn.Module):
    """Input (b, *grid, in_ch) -> output (b, *grid, out_ch). Coordinates are appended."""

    def __init__(self, modes, width=64, layers=4, in_ch=1, out_ch=1, proj=128, pad=0):
        super().__init__()
        self.pad = pad
        self.dim = len(modes)
        self.width = width
        self.lift = nn.Linear(in_ch + self.dim, width)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(layers))
        conv = nn.Conv1d if self.dim == 1 else nn.Conv2d
        self.local = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, out_ch))

    @staticmethod
    def grid(x):
        s = x.shape[1:-1]
        axes = [torch.linspace(0, 1, n, device=x.device, dtype=x.dtype) for n in s]
        g = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
        return g.expand(x.shape[0], *g.shape)

    def features(self, x):
        h = self.lift(torch.cat([x, self.grid(x)], -1)).movedim(-1, 1)
        if self.pad:
            h = F.pad(h, [0, self.pad] * self.dim)      # non-periodic domains
        for i, (s, l) in enumerate(zip(self.spectral, self.local)):
            h = s(h) + l(h)
            if i < len(self.spectral) - 1:
                h = F.gelu(h)
        if self.pad:
            h = h[..., :-self.pad] if self.dim == 1 else h[..., :-self.pad, :-self.pad]
        return h.movedim(1, -1)

    def forward(self, x):
        return self.proj(self.features(x))
