import torch
import numpy as np
import operator
from functools import reduce


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def count_params(model) -> int:
    c = 0
    for p in model.parameters():
        sz = list(p.size() + (2,) if p.is_complex() else p.size())
        c += reduce(operator.mul, sz, 1)
    return c
