"""Loss functions for Step 1 validation.

DeterministicPowerLawLoss: power-law w(sigma) = sigma^beta with weight normalization.
GeneralWeightingLoss: arbitrary w(sigma) from get_weighting() for named schemes.

Both use deterministic sigma integration (K=50-100 fixed log-spaced sigma values per step).
The inner loop calls net(x_noisy, sigma) generically, so both LinearDenoiserShared and
MLPDenoiser are supported without modification.

Vectorization: noisy inputs are stacked as [K*N, d] and passed in one forward call.
Each of the K sigma values is repeated N times to match samples.
"""

import sys
import os
from math import log10

import numpy as np
import torch

from .config import K_SIGMA, SIGMA_0, SIGMA_T


class DeterministicPowerLawLoss:
    """Weighted denoising loss with w(sigma) = sigma^beta.

    Weight normalization ensures lr=0.01 is stable for all beta values:
        w_norm(sigma_j) = sigma_j^beta / A_max
    where A_max = mean(w * (lambda_max + sigma^2)), so A_1 = 1.

    Args:
        beta: power-law exponent
        K_sigma: number of sigma grid points
        sigma_min, sigma_max: noise bounds
        lambda_max: largest eigenvalue (for normalization)
    """

    def __init__(self, beta, K_sigma=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T,
                 lambda_max=1.0):
        self.beta = beta
        self.sigmas = torch.logspace(log10(sigma_min), log10(sigma_max), K_sigma)
        raw_weights = self.sigmas ** beta
        A_max = (raw_weights * (lambda_max + self.sigmas**2)).mean()
        self.weights = raw_weights / A_max
        self.normalization_factor = float(A_max)

    def __call__(self, net, X):
        K = len(self.sigmas)
        N, d = X.shape
        device = X.device

        sigmas = self.sigmas.to(device)
        weights = self.weights.to(device)

        # Build [K, N, d] noisy inputs
        X_exp   = X.unsqueeze(0).expand(K, -1, -1)                    # [K, N, d]
        noise   = torch.randn(K, N, d, device=device) * sigmas.view(K, 1, 1)
        X_noisy = X_exp + noise                                        # [K, N, d]

        # Flatten to [K*N, d] and build matching sigma vector [K*N]
        # then call net generically — works for LinearDenoiserShared and MLPDenoiser
        x_flat     = X_noisy.reshape(K * N, d)
        sigma_flat = sigmas.view(K, 1).expand(K, N).reshape(K * N)    # [K*N]
        D_out      = net(x_flat, sigma_flat).reshape(K, N, d)         # [K, N, d]

        # Per-sigma MSE, averaged over samples and dimensions
        mse = ((D_out - X_exp) ** 2).sum(dim=-1).mean(dim=-1)         # [K]

        return (weights * mse).mean()


class GeneralWeightingLoss:
    """Weighted denoising loss with arbitrary w(sigma) from get_weighting().

    For named schemes (EDM, min-SNR, P2). Evaluates w_fn on a fixed sigma grid
    and normalizes identically to DeterministicPowerLawLoss.

    Args:
        w_fn: callable sigma -> w(sigma) (the eta_fn from get_weighting())
        K_sigma: number of sigma grid points
        sigma_min, sigma_max: noise bounds
        lambda_max: largest eigenvalue (for normalization)
    """

    def __init__(self, w_fn, K_sigma=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T,
                 lambda_max=1.0):
        self.sigmas = torch.logspace(log10(sigma_min), log10(sigma_max), K_sigma)
        raw_weights = torch.tensor([w_fn(float(s)) for s in self.sigmas],
                                   dtype=torch.float32)
        A_max = (raw_weights * (lambda_max + self.sigmas**2)).mean()
        self.weights = raw_weights / A_max
        self.normalization_factor = float(A_max)

    def __call__(self, net, X):
        K = len(self.sigmas)
        N, d = X.shape
        device = X.device

        sigmas = self.sigmas.to(device)
        weights = self.weights.to(device)

        X_exp = X.unsqueeze(0).expand(K, -1, -1)
        noise = torch.randn(K, N, d, device=device) * sigmas.view(K, 1, 1)
        X_noisy = X_exp + noise

        x_flat     = X_noisy.reshape(K * N, d)
        sigma_flat = sigmas.view(K, 1).expand(K, N).reshape(K * N)
        D_out      = net(x_flat, sigma_flat).reshape(K, N, d)

        mse = ((D_out - X_exp) ** 2).sum(dim=-1).mean(dim=-1)

        return (weights * mse).mean()
