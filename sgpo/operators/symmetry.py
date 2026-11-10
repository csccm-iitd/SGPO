"""Exact symmetry groups used for augmentation and group-averaged prediction."""
import torch


def d4(t, k):
    """k-th element (0..7) of the dihedral group acting on the grid axes of (b, s, s, c)."""
    if k & 4:
        t = t.transpose(1, 2)
    return torch.rot90(t, k & 3, (1, 2))


def d4_inv(t, k):
    t = torch.rot90(t, -(k & 3), (1, 2))
    return t.transpose(1, 2) if k & 4 else t


def group_average(f, x):
    """Projects f onto D4-equivariant maps: mean over g of g^-1 f(g x)."""
    return sum(d4_inv(f(d4(x, k)), k) for k in range(8)) / 8
