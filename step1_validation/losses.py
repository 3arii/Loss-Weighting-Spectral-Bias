"""Loss functions for Step 1 validation."""

from math import exp, log10
import torch

from .config import K_SIGMA, SIGMA_0, SIGMA_T


def _build_weights(sigmas, beta, w_max=None, normalize="mean"):
    """Power-law w(sigma) = sigma^beta, with optional clipping and normalization.

    normalize='mean' -> mean(w) = 1. Keeps effective learning rate constant
    across beta (Binxu's meeting-2 concern: unnormalized weights conflate with
    LR and change SGD stability). normalize='rms' keeps Σw² constant instead.
    """
    w = sigmas ** beta
    if w_max is not None:
        w = torch.clamp(w, 1.0 / w_max, w_max)
    if normalize == "mean":
        w = w / w.mean()
    elif normalize == "rms":
        w = w / torch.sqrt((w ** 2).mean())
    elif normalize is None or normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return w


class PerSigmaPowerLawLoss:
    """Per-sigma loss: each sigma_j has its own independent denoiser.

    This matches the theory exactly. The loss is:
        L = (1/K) * sum_j  w(sigma_j)/A_max * E[ || D_j(x + sigma_j*z) - x ||^2 ]

    Each D_j has its own parameters (independent across sigma).
    A_max = mean(w(sigma_j)*(lambda_max+sigma_j^2)) normalizes so that the SGD
    stability limit is ~O(1) across all beta values (not beta-dependent).
    Step-to-theory-time conversion: tau = step * lr / (K * A_max).
    """

    def __init__(self, beta, K_sigma=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T,
                 lambda_max=1.0, w_max=None):
        self.beta = beta
        self.K_sigma = K_sigma
        self.w_max = w_max
        self.sigmas = torch.logspace(log10(sigma_min), log10(sigma_max), K_sigma)
        raw_weights = self.sigmas ** beta
        if w_max is not None:
            raw_weights = torch.clamp(raw_weights, 1.0 / w_max, w_max)
        A_max = (raw_weights * (lambda_max + self.sigmas ** 2)).mean()
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


class SharedMLPPowerLawLoss:
    """Joint w(sigma)-weighted denoising loss for a shared MLP across sigma.

    Standard diffusion-training recipe: for each example in the minibatch,
    draw a sigma from the log-uniform grid, add Gaussian noise, and minimize
        L = E_{x, j, z} [ w(sigma_j) * || D(x + sigma_j * z, sigma_j) - x ||^2 ]

    Weights are mean-normalized by default so that changing beta does not
    shift the effective learning rate (meeting-2 guidance from Binxu).
    """

    def __init__(self, beta, K_sigma=K_SIGMA, sigma_min=SIGMA_0,
                 sigma_max=SIGMA_T, w_max=None, normalize="mean"):
        self.beta = beta
        self.K_sigma = K_sigma
        self.w_max = w_max
        self.normalize = normalize
        self.sigmas = torch.logspace(log10(sigma_min), log10(sigma_max), K_sigma)
        self.weights = _build_weights(self.sigmas, beta, w_max=w_max,
                                      normalize=normalize)

    def __call__(self, model, X_batch):
        """X_batch: [N, d] clean samples."""
        N, d = X_batch.shape
        device = X_batch.device
        sigmas = self.sigmas.to(device)
        weights = self.weights.to(device)

        idx = torch.randint(0, self.K_sigma, (N,), device=device)
        sig = sigmas[idx]                    # [N]
        w = weights[idx]                     # [N]

        noise = torch.randn(N, d, device=device) * sig[:, None]
        X_noisy = X_batch + noise
        D_out = model(X_noisy, sig)          # [N, d]

        per_sample_mse = ((D_out - X_batch) ** 2).sum(dim=-1)  # [N]
        return (w * per_sample_mse).mean()


class BetaPowerEDMLoss:
    """Binxu's EDMLoss with the Karras weight swapped for a power-law w(σ) = σ^β.

    Mirrors `core/diffusion_edm_lib.py::EDMLoss` in DiffusionLearningCurve:
      - σ ~ LogNormal(P_mean, P_std²) (continuous, one σ per sample per step)
      - σ_data-free weight, because σ^β is what we want to sweep over
      - x0 parameterization: loss = w(σ)·‖D(x + σ·ε, σ) - x‖²

    The mean-normalization `w / E_LN[σ^β]` makes the effective learning rate
    β-invariant (Binxu meeting-2 concern: unnormalized w confounds β with LR,
    so β-sweep would also be an effective-LR sweep).

    Key difference from SharedMLPPowerLawLoss:
        old: σ sampled from a discrete K-point log-uniform grid → effective
             weighting is σ^(β-1)/C (extra 1/σ from log-uniform density)
        new: σ sampled continuously from LogNormal, matching Binxu's recipe
             → effective weighting is w(σ) = σ^β exactly
    """

    def __init__(self, beta, P_mean=-1.2, P_std=1.2, normalize=True):
        self.beta = beta
        self.P_mean = P_mean
        self.P_std = P_std
        self.normalize = normalize
        self.E_w = (exp(beta * P_mean + beta ** 2 * P_std ** 2 / 2.0)
                    if normalize else 1.0)

    def __call__(self, model, X):
        """X: [N, d] clean samples. Returns scalar loss."""
        N = X.shape[0]
        device = X.device
        rnd = torch.randn(N, device=device)
        sigma = (rnd * self.P_std + self.P_mean).exp()        # [N]  LogNormal
        weight = (sigma ** self.beta) / self.E_w              # [N]
        noise = torch.randn_like(X) * sigma[:, None]
        D = model(X + noise, sigma)                           # [N, d]
        per_sample_mse = ((D - X) ** 2).sum(dim=-1)           # [N]
        return (weight * per_sample_mse).mean()
