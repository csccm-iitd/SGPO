"""Mean functions for GP operator learning.

Provides neural-network and basis-function driven mean functions
that go beyond the constant/zero/linear means in GPyTorch.
"""

from sgpo.means.nn_mean import NNMean
from sgpo.means.basis_mean import FourierBasisMean, PolynomialBasisMean
from sgpo.wno.mean import WNOMean

__all__ = [
    "NNMean",
    "FourierBasisMean",
    "PolynomialBasisMean",
    "WNOMean",
]
