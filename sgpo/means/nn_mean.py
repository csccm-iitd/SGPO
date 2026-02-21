"""Neural-network mean function for GP models.

    m(x) = NN(x)

A learnable, flexible prior mean that captures the dominant
input-output mapping.  The GP then models the residual:

"""

import torch
import torch.nn as nn
import gpytorch


class NNMean(gpytorch.means.Mean):
    """Neural-network mean function.

    Wraps an arbitrary ``nn.Module`` as a GPyTorch mean.
    The module must map ``(batch, d_in) → (batch, d_out)``
    (or ``(batch, d_in) → (batch,)`` for single-output).

    Args:
        nn_module: Neural network to use as the mean function.
            If ``None``, a default 3-layer MLP is constructed.
        input_dim: Required if ``nn_module`` is ``None``.
        output_dim: Required if ``nn_module`` is ``None``.
        hidden_dim: Hidden layer width for the default MLP.
        num_hidden: Number of hidden layers for the default MLP.
    """

    def __init__(
        self,
        nn_module: nn.Module | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
        hidden_dim: int = 128,
        num_hidden: int = 3,
    ):
        super().__init__()

        if nn_module is not None:
            self.nn = nn_module
        else:
            if input_dim is None or output_dim is None:
                raise ValueError(
                    "Must provide input_dim and output_dim if nn_module is None."
                )
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            for _ in range(num_hidden - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
