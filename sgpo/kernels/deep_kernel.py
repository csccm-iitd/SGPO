"""Deep Kernel Learning (DKL) 

Wilson et al., "Deep Kernel Learning" (AISTATS 2016).

This is the generic version; for the WNO-specific variant see
`sgpo.wno.embedding.WNOEmbedding` + the model's `wno_embedding`
argument.
"""

import torch
import torch.nn as nn
from gpytorch.kernels import Kernel


class DeepKernel(Kernel):
    """Deep Kernel: base_kernel ∘ feature_extractor.

    Any ``nn.Module`` that maps (batch, d_in) -> (batch, d_out)
    can be used as the feature extractor.  The base kernel then
    operates in the d_out-dimensional latent space.

    Args:
        feature_extractor: ``nn.Module`` mapping inputs to features.
        base_kernel: GPyTorch ``Kernel`` applied in feature space.
    """

    has_lengthscale = False

    def __init__(self, feature_extractor: nn.Module, base_kernel: Kernel,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        z1 = self.feature_extractor(x1)
        z2 = self.feature_extractor(x2)
        return self.base_kernel.forward(z1, z2, diag=diag, **params)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
