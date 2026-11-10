"""Run an operator trained on a coarse periodic grid on fine-grid inputs, by exact band-limited resampling.

For band-limited fields on the torus (e.g. NS Re=500 at 1024^2, where output energy above k=32 is 1e-5 of the total),
point-sampling the input every `factor` points loses nothing below the coarse Nyquist frequency, and zero-padding the
Fourier coefficients of the coarse features interpolates them back to the fine grid exactly for band-limited content.
The GP layer of LoGoS-GPO then works on the fine grid (one value per fine grid point).
"""
import torch
import torch.nn as nn


def fourier_upsample(h, factor):
    """h (b, c, n, n) periodic -> (b, c, n*factor, n*factor) by zero-padding the spectrum (keeps the mean and amplitudes)."""
    n = h.shape[-1]; m = n * factor
    F = torch.fft.rfft2(h)
    out = torch.zeros(*h.shape[:-2], m, m // 2 + 1, dtype=F.dtype, device=h.device)
    k = n // 2
    out[..., :k, :k + 1] = F[..., :k, :k + 1]
    out[..., -k:, :k + 1] = F[..., -k:, :k + 1]
    return torch.fft.irfft2(out, s=(m, m)) * factor ** 2


class Resample(nn.Module):
    def __init__(self, op, factor):
        super().__init__()
        self.op, self.factor = op, factor
        self.width = op.width

    def features(self, x):                                       # x (b, N, N, c) fine grid, N = n * factor
        h = self.op.features(x[:, ::self.factor, ::self.factor])  # (b, n, n, w) on the training grid
        return fourier_upsample(h.movedim(-1, 1), self.factor).movedim(1, -1)

    def proj(self, h):
        return self.op.proj(h)

    def forward(self, x):
        return self.proj(self.features(x))
