import torch
import numpy as np
from sgpo.data.readers import MatReader
from sgpo.data.normalizers import UnitGaussianNormalizer


def load_burger(cfg, device):
    """Load Burgers equation data from .mat file.
    Fields: cfg.x_field (input a(x)), cfg.y_field (output u(x)).
    1D spatial, subsampled by cfg.sub.
    """
    reader = MatReader(cfg.train_path)
    x_data = reader.read_field(cfg.x_field)[:, ::cfg.sub]
    y_data = reader.read_field(cfg.y_field)[:, ::cfg.sub]

    x_train = x_data[:cfg.ntrain, :]
    y_train = y_data[:cfg.ntrain, :]
    x_test = x_data[-cfg.ntest:, :]
    y_test = y_data[-cfg.ntest:, :]

    x_normalizer, y_normalizer = None, None
    if cfg.normalize_x:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if cfg.normalize_y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    # Flatten for GP: (n, d)
    x_tr = x_train.reshape(cfg.ntrain, -1).to(device)
    y_tr = y_train.reshape(cfg.ntrain, -1).to(device)
    x_t = x_test.reshape(cfg.ntest, -1).to(device)
    y_t = y_test.reshape(cfg.ntest, -1).to(device)

    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer


def load_darcy(cfg, device):
    """Load 2D Darcy flow data from .mat file(s).
    Separate train/test files. Supports boundary_zero.
    """
    r = cfg.sub
    s = cfg.resolution[0]

    reader = MatReader(cfg.train_path)
    x_train = reader.read_field(cfg.x_field)[:cfg.ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field(cfg.y_field)[:cfg.ntrain, ::r, ::r][:, :s, :s]

    if cfg.boundary_zero:
        y_train[:, 0, :] = 0
        y_train[:, -1, :] = 0
        y_train[:, :, 0] = 0
        y_train[:, :, -1] = 0

    # Load test data from separate file
    if cfg.test_path:
        reader.load_file(cfg.test_path)
    x_test = reader.read_field(cfg.x_field)[:cfg.ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field(cfg.y_field)[:cfg.ntest, ::r, ::r][:, :s, :s]

    if cfg.boundary_zero:
        y_test[:, 0, :] = 0
        y_test[:, -1, :] = 0
        y_test[:, :, 0] = 0
        y_test[:, :, -1] = 0

    x_normalizer, y_normalizer = None, None
    if cfg.normalize_x:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if cfg.normalize_y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    # Flatten for GP: (n, s*s)
    x_tr = x_train.reshape(cfg.ntrain, -1).to(device)
    y_tr = y_train.reshape(cfg.ntrain, -1).to(device)
    x_t = x_test.reshape(cfg.ntest, -1).to(device)
    y_t = y_test.reshape(cfg.ntest, -1).to(device)

    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer


def load_darcy_notch(cfg, device):
    """Load 2D Darcy triangular notch data from .mat file.
    Fields: boundCoeff (input), sol (output). Single file, train/test split.
    """
    r = cfg.sub
    s = cfg.resolution[0]

    reader = MatReader(cfg.train_path)
    x_train = reader.read_field(cfg.x_field)[:cfg.ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field(cfg.y_field)[:cfg.ntrain, ::r, ::r][:, :s, :s]

    x_test = reader.read_field(cfg.x_field)[-cfg.ntest:, ::r, ::r][:, :s, :s]
    y_test = reader.read_field(cfg.y_field)[-cfg.ntest:, ::r, ::r][:, :s, :s]

    x_normalizer, y_normalizer = None, None
    if cfg.normalize_x:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if cfg.normalize_y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    # Flatten for GP
    x_tr = x_train.reshape(cfg.ntrain, -1).to(device)
    y_tr = y_train.reshape(cfg.ntrain, -1).to(device)
    x_t = x_test.reshape(cfg.ntest, -1).to(device)
    y_t = y_test.reshape(cfg.ntest, -1).to(device)

    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer


def load_navier(cfg, device):
    """Load 2D Navier-Stokes data from .npy files.
    Separate input/output .npy files. Shape: (x, y, n) -> permuted to (n, x, y).
    """
    r = cfg.sub
    s = cfg.resolution[0]

    reader_input = np.load(cfg.train_path)   # (x, y, n)
    reader_output = np.load(cfg.test_path)   # (x, y, n)

    reader_input = torch.tensor(reader_input).permute(2, 1, 0).float()   # (n, y, x)
    reader_output = torch.tensor(reader_output).permute(2, 1, 0).float()

    x_train = reader_input[:cfg.ntrain, ::r, ::r][:, :s, :s]
    y_train = reader_output[:cfg.ntrain, ::r, ::r][:, :s, :s]

    x_test = reader_input[cfg.ntrain:cfg.ntrain + cfg.ntest, ::r, ::r][:, :s, :s]
    y_test = reader_output[cfg.ntrain:cfg.ntrain + cfg.ntest, ::r, ::r][:, :s, :s]

    x_normalizer, y_normalizer = None, None
    if cfg.normalize_x:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if cfg.normalize_y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    # Flatten for GP
    x_tr = x_train.reshape(cfg.ntrain, -1).to(device)
    y_tr = y_train.reshape(cfg.ntrain, -1).to(device)
    x_t = x_test.reshape(cfg.ntest, -1).to(device)
    y_t = y_test.reshape(cfg.ntest, -1).to(device)

    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer


def load_wave(cfg, device):
    """Load 1D wave equation data from .npz files.
    Contains x, t, u arrays. Input: u[:, 0, :] (IC), output: u[:, t_step, :].
    """
    data_train = np.load(cfg.train_path)
    data_test = np.load(cfg.test_path)
    x_train_raw, u_train = data_train["x"], data_train["u"]
    x_test_raw, u_test = data_test["x"], data_test["u"]

    # Default: input at t=0, output at t=80
    t_in = 0
    t_out = 80
    x_data_train = torch.tensor(u_train[:, t_in, :]).float()
    y_data_train = torch.tensor(u_train[:, t_out, :]).float()
    x_data_test = torch.tensor(u_test[:, t_in, :]).float()
    y_data_test = torch.tensor(u_test[:, t_out, :]).float()

    x_train = x_data_train[:cfg.ntrain, :]
    y_train = y_data_train[:cfg.ntrain, :]
    x_test = x_data_test[:cfg.ntest, :]
    y_test = y_data_test[:cfg.ntest, :]

    x_normalizer, y_normalizer = None, None
    if cfg.normalize_x:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if cfg.normalize_y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    # Flatten for GP (already 1D, shape: n x size)
    x_tr = x_train.reshape(cfg.ntrain, -1).to(device)
    y_tr = y_train.reshape(cfg.ntrain, -1).to(device)
    x_t = x_test.reshape(cfg.ntest, -1).to(device)
    y_t = y_test.reshape(cfg.ntest, -1).to(device)

    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer


LOADERS = {
    "burger": load_burger,
    "darcy": load_darcy,
    "darcy_notch": load_darcy_notch,
    "navier": load_navier,
    "wave": load_wave,
}


def load_data(cfg, device):
    """Load data for the experiment specified by cfg.name."""
    if cfg.name not in LOADERS:
        raise ValueError(
            f"Unknown experiment: {cfg.name}. "
            f"Available: {list(LOADERS.keys())}"
        )
    return LOADERS[cfg.name](cfg, device)
