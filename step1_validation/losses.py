"""Loss functions for Step 1 validation."""

from math import log10
import torch

from .config import K_SIGMA, SIGMA_0, SIGMA_T


class PerSigmaPowerLawLoss:
    """Per-sigma loss: each sigma_j has its own independent denoiser.

    This matches the theory exactly. The loss is:
        L = (1/K) * sum_j  w(sigma_j) * E[ || D_j(x + sigma_j*z) - x ||^2 ]

    Each D_j has its own parameters (independent across sigma).
    Weight normalization ensures lr stability across all beta.
    """

    def __init__(self, beta, K_sigma=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T,
                 lambda_max=1.0):
        self.beta = beta
        self.K_sigma = K_sigma
        self.sigmas = torch.logspace(log10(sigma_min), log10(sigma_max), K_sigma)
        raw_weights = self.sigmas ** beta
        A_max = (raw_weights * (lambda_max + self.sigmas**2)).mean()
        self.weights = raw_weights / A_max
        self.normalization_factor = float(A_max)

    def __call__(self, model, X):
        """model: LinearDenoiserPerSigma, X: [N, d]"""
        N, d = X.shape
        device = X.device
        sigmas = self.sigmas.to(device)
        weights = self.weights.to(device)

        total_loss = torch.tensor(0.0, device=device)
        for j in range(self.K_sigma):
            noise = torch.randn(N, d, device=device) * sigmas[j]
            X_noisy = X + noise
            D_out = model.forward_at_sigma_idx(X_noisy, j)  # [N, d]
            mse = ((D_out - X)**2).sum(dim=-1).mean()  # scalar
            total_loss = total_loss + weights[j] * mse

        return total_loss / self.K_sigma
