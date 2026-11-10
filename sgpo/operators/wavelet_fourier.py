"""Multi-level wavelet-Fourier convolution (Eq. 30), used by EWNO.

Every decomposition level and detail band of the DWT gets its own Fourier weights, with up to `modes` frequencies
per level, so the layer acts on all scales (the WaveConv2d of the original WNO learns weights on the coarsest
level only).
"""
import torch
import torch.nn as nn
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D


def _cweights(cin, cout, *m):
    return nn.Parameter(torch.randn(cin, cout, *m, dtype=torch.cfloat) / (cin * cout) ** 0.5 / cin ** 0.5)


class WaveConv2dML(nn.Module):
    def __init__(self, width, level, size, wavelet="db4", modes=12, mode="symmetric"):
        super().__init__()
        self.dwt, self.idwt = DWT(J=level, mode=mode, wave=wavelet), IDWT(mode=mode, wave=wavelet)
        lo, hi = self.dwt(torch.zeros(1, 1, *size))
        shapes = [lo.shape[-2:]] + [h.shape[-2:] for h in hi]                   # coarse, then fine..coarse
        self.m = [(min(modes, s[0] // 2), min(modes, s[1] // 2 + 1)) for s in shapes]
        n_bands = [1] + [3] * level
        self.w = nn.ParameterList()
        for (m1, m2), nb in zip(self.m, n_bands):
            for _ in range(nb * 2):                                              # two corners per band
                self.w.append(_cweights(width, width, m1, m2))

    def _fconv(self, x, w1, w2, m):
        m1, m2 = m
        xf = torch.fft.rfft2(x)
        out = torch.zeros(x.shape[0], w1.shape[1], *xf.shape[-2:], dtype=torch.cfloat, device=x.device)
        out[..., :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[..., :m1, :m2], w1)
        out[..., -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[..., -m1:, :m2], w2)
        return torch.fft.irfft2(out, s=x.shape[-2:])

    def forward(self, x):
        lo, hi = self.dwt(x)
        it = iter(self.w)
        lo = self._fconv(lo, next(it), next(it), self.m[0])
        out_hi = []
        for h, m in zip(hi, self.m[1:]):
            out_hi.append(torch.stack([self._fconv(h[:, :, b], next(it), next(it), m) for b in range(3)], 2))
        return self.idwt((lo, out_hi))


class WaveConv1dML(nn.Module):
    def __init__(self, width, level, size, wavelet="db4", modes=16, mode="symmetric"):
        super().__init__()
        self.dwt, self.idwt = DWT1D(J=level, mode=mode, wave=wavelet), IDWT1D(mode=mode, wave=wavelet)
        lo, hi = self.dwt(torch.zeros(1, 1, size))
        self.m = [min(modes, s.shape[-1] // 2 + 1) for s in [lo] + hi]
        self.w = nn.ParameterList(_cweights(width, width, m) for m in self.m)

    def _fconv(self, x, w, m):
        xf = torch.fft.rfft(x)
        out = torch.zeros(x.shape[0], w.shape[1], xf.shape[-1], dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bix,iox->box", xf[..., :m], w)
        return torch.fft.irfft(out, n=x.shape[-1])

    def forward(self, x):
        lo, hi = self.dwt(x)
        lo = self._fconv(lo, self.w[0], self.m[0])
        hi = [self._fconv(h, w, m) for h, w, m in zip(hi, self.w[1:], self.m[1:])]
        return self.idwt((lo, hi))
