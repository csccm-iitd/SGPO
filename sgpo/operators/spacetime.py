"""Space-time wrapper: a 3D operator over (space, time) for problems whose input is static in time.

Input (b, *grid, C) static channels -> repeated over the T snapshot times with the time value appended as a channel
(as in the U-FNO CO2 data, whose time channel is the only time-varying input) -> 3D operator over (*grid, T) ->
output (b, *grid, T). features/proj keep the operator's interface, so LoGoS-GPO sees one value per (x, t).
"""
import torch
import torch.nn as nn


class SpaceTime(nn.Module):
    def __init__(self, op, times):
        super().__init__()
        self.op = op
        self.register_buffer("times", torch.as_tensor(times, dtype=torch.float32))
        self.width = op.width

    def lift_time(self, x):
        b, *g, c = x.shape
        t = self.times.view(1, *[1] * len(g), -1, 1).expand(b, *g, -1, 1)
        return torch.cat([x[..., None, :].expand(b, *g, len(self.times), c), t], -1)

    def features(self, x):
        return self.op.features(self.lift_time(x))              # (b, *grid, T, width)

    def proj(self, h):
        return self.op.proj(h)[..., 0]                           # (b, *grid, T)

    def forward(self, x):
        return self.proj(self.features(x))
