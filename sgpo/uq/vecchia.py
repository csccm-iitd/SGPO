"""Neighbour sets of the Vecchia approximation: each site conditions on its k nearest predecessors in a fixed ordering."""
import numpy as np
from scipy.spatial import cKDTree


def predecessor_neighbours(c, k, oversample=8):
    """For points c (d, D) already in the chosen order, the k nearest among c[:i] for every i (-1 padded).

    A KD-tree query of oversample*k candidates is filtered to predecessors; sites left short (only early
    ones in a random order) fall back to an exact search over their predecessors.
    """
    d = len(c)
    q = min(d, oversample * k + 1)
    _, cand = cKDTree(c).query(c, k=q)
    cand = cand.reshape(d, q)
    pred = cand < np.arange(d)[:, None]
    first = np.argsort(~pred, axis=1, kind="stable")[:, :k]        # predecessor candidates, nearest first
    take = np.take_along_axis(pred, first, 1)
    nbr = np.where(take, np.take_along_axis(cand, first, 1), -1)
    short = np.nonzero(take.sum(1) < np.minimum(k, np.arange(d)))[0]
    for i in short:
        dist = ((c[:i] - c[i]) ** 2).sum(-1)
        j = np.argsort(dist, kind="stable")[: min(k, i)]
        nbr[i, : len(j)] = j
    return nbr
