"""Power-law loss with weight normalization for Step 1 validation."""

from math import log10
import torch

from .config import K_SIGMA, SIGMA_0, SIGMA_T


class DeterministicPowerLawLoss:
    """w(sigma) = sigma^beta, normalized so lr=0.01 is stable for all beta.

    Evaluates loss at K fixed log-spaced sigma values per step.
    All K*N forward passes in one batched call.
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

        X_exp = X.unsqueeze(0).expand(K, -1, -1)                       # [K, N, d]
        noise = torch.randn(K, N, d, device=device) * sigmas.view(K, 1, 1)
        X_noisy = X_exp + noise                                         # [K, N, d]

        # Single batched matmul: flatten to [K*N, d]
        D_out = X_noisy.reshape(K * N, d) @ net.W.T + net.b            # [K*N, d]
        D_out = D_out.reshape(K, N, d)

        mse = ((D_out - X_exp)**2).sum(dim=-1).mean(dim=-1)            # [K]
        return (weights * mse).mean()
