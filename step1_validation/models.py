"""Denoiser models for Step 1 validation."""

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


class LinearDenoiserPerSigma(nn.Module):
    """Independent diagonal denoiser at each sigma level.

    For each sigma_j, the denoiser is: D_j(x) = diag(a_k[j]) * x
    where a_k[j] is a learnable scalar per (sigma_j, mode_k) pair.

    This matches the theory setup exactly: independent W_sigma at each noise level.
    With diagonal covariance, we only need the diagonal (d scalars per sigma).

    Args:
        ndim: data dimensionality (d)
        K_sigma: number of sigma grid points
    """

    def __init__(self, ndim, K_sigma):
        super().__init__()
        # a_k[j, k] = projection of W_sigma_j onto eigenmode k
        # Shape: [K_sigma, ndim]
        self.a_k = nn.Parameter(torch.zeros(K_sigma, ndim))
        self.K_sigma = K_sigma
        self.ndim = ndim

    def forward_at_sigma_idx(self, x, sigma_idx):
        """Denoise x using the parameters for sigma index sigma_idx.
        x: [N, d], sigma_idx: int -> returns [N, d]"""
        return x * self.a_k[sigma_idx]  # element-wise: diag(a_k[j]) * x
