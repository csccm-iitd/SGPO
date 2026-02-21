import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sgpo.wno.wavelet_conv import WaveConv1d, WaveConv2d


class WNO1d(nn.Module):
    """1D Wavelet Neural Operator.

    Architecture: lift -> [WaveConv1d + Conv1d] x layers -> project
    v(x) -> fc0 -> {WaveConv(v) + W(v), activation} x L -> fc1 -> fc2 -> u(x)

    Input:  (batch, size, 1) or (batch, size) -- flattened input function
    Output: (batch, size, 1) -- predicted output function
    """

    def __init__(self, width, level, layers, size, wavelet="db4",
                 in_channel=2, grid_range=1, padding=0):
        super().__init__()
        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = padding

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        self.fc0 = nn.Linear(self.in_channel, self.width)
        for _ in range(self.layers):
            self.conv.append(WaveConv1d(self.width, self.width, self.level,
                                        self.size, self.wavelet))
            self.w.append(nn.Conv1d(self.width, self.width, 1))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, size) -- 2D flattened input
        x = x.unsqueeze(-1)  # (batch, size, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)      # (batch, size, in_channel)
        x = self.fc0(x)                        # (batch, size, width)
        x = x.permute(0, 2, 1)                # (batch, width, size)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding])

        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)                # (batch, size, width)
        x = F.mish(self.fc1(x))               # (batch, size, 128)
        x = self.fc2(x)                        # (batch, size, 1)
        return x.squeeze(-1)                   # (batch, size)

    def get_features(self, x):
        """Extract intermediate features after wavelet layers, before fc projection.
        Returns (batch, size * width) -- flattened spatial features.
        """
        x = x.unsqueeze(-1)  # (batch, size, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding])

        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding]
        # x is (batch, width, size) -- flatten to (batch, width * size)
        return x.reshape(x.shape[0], -1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x),
                             dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class WNO2d(nn.Module):
    """2D Wavelet Neural Operator.

    Architecture: lift -> [WaveConv2d + Conv2d] x layers -> project
    v(x,y) -> fc0 -> {WaveConv(v) + W(v), activation} x L -> fc1 -> fc2 -> u(x,y)

    Input:  (batch, sx*sy) -- flattened 2D input, or (batch, sx, sy, 1)
    Output: (batch, sx*sy) -- flattened 2D output
    """

    def __init__(self, width, level, layers, size, wavelet="db4",
                 in_channel=3, grid_range=None, padding=0):
        super().__init__()
        if grid_range is None:
            grid_range = [1, 1]
        self.level = level
        self.width = width
        self.layers = layers
        self.size = size  # [sx, sy]
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = padding

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        self.fc0 = nn.Linear(self.in_channel, self.width)
        for _ in range(self.layers):
            self.conv.append(WaveConv2d(self.width, self.width, self.level,
                                        self.size, self.wavelet))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, sx*sy) -- 2D flattened input
        sx, sy = self.size[0], self.size[1]
        x = x.reshape(x.shape[0], sx, sy, 1).float()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)      # (batch, sx, sy, in_channel)
        x = self.fc0(x)                        # (batch, sx, sy, width)
        x = x.permute(0, 3, 1, 2)             # (batch, width, sx, sy)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)             # (batch, sx, sy, width)
        x = F.mish(self.fc1(x))               # (batch, sx, sy, 128)
        x = self.fc2(x)                        # (batch, sx, sy, 1)
        # Flatten output: (batch, sx*sy)
        return x.reshape(x.shape[0], -1)

    def get_features(self, x):
        """Extract intermediate features after wavelet layers, before fc projection.
        Returns (batch, width * sx * sy) -- flattened spatial feature maps.
        """
        sx, sy = self.size[0], self.size[1]
        x = x.reshape(x.shape[0], sx, sy, 1).float()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        # x is (batch, width, sx, sy) -- flatten
        return x.reshape(x.shape[0], -1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x),
                             dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y),
                             dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
