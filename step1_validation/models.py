"""Linear denoiser for Step 1 validation."""

import torch
import torch.nn as nn


class LinearDenoiserShared(nn.Module):
    """D(x; sigma) = W*x + b, shared across all sigma.
    Initialized at W=0, b=0. Sigma is accepted but ignored."""

    def __init__(self, ndim):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(ndim, ndim))
        self.b = nn.Parameter(torch.zeros(ndim))

    def forward(self, x, sigma, cond=None):
        return x @ self.W.T + self.b
