"""Denoiser models for Step 1 validation.

LinearDenoiserShared: single shared W matrix (sigma-blind baseline).
MLPDenoiser: sigma-conditioned MLP using UNetBlockStyleMLP_backbone from
    DiffusionLearningCurve, with deterministic K-sigma training.
"""

import sys
import torch
import torch.nn as nn

# Import backbone from DiffusionLearningCurve
_DLC_PATH = "/n/home12/binxuwang/Github/DiffusionLearningCurve"
if _DLC_PATH not in sys.path:
    sys.path.insert(0, _DLC_PATH)
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone  # noqa: E402


class LinearDenoiserShared(nn.Module):
    """D(x; sigma) = W*x + b, shared across all sigma. No preconditioning.

    Initialized at W=0, b=0 so a_k(0)=0 for all eigenmodes.
    Accepts sigma in forward() for API compatibility but ignores it.
    """

    def __init__(self, ndim):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(ndim, ndim))
        self.b = nn.Parameter(torch.zeros(ndim))

    def forward(self, x, sigma, cond=None):
        return x @ self.W.T + self.b


class MLPDenoiser(nn.Module):
    """Sigma-conditioned MLP denoiser. No EDM preconditioning.

    Architecture:
        x_in  = x_noisy / sqrt(1 + sigma^2)        # simple input scaling
        t_enc = log(sigma) / 4                      # scalar noise embedding
        out   = UNetBlockStyleMLP_backbone(x_in, t_enc)
              = Linear(d→nhidden)
              + nlayers×UNetMLPBlock(nhidden, nhidden, time_embed_dim)
                  [each block: LN→fc→SiLU + adaptive scale+shift from t_embed + residual]
              + Linear(nhidden→d)

    The network outputs a direct denoised prediction D(x_noisy, sigma) ≈ x_clean.
    sigma is embedded via GaussianFourierProjection inside UNetBlockStyleMLP_backbone.

    Args:
        ndim:           data dimensionality
        nhidden:        hidden layer width (default 256)
        nlayers:        total number of layers incl. in/out projections (default 5)
        time_embed_dim: Fourier feature dimension for sigma embedding (default 64)
    """

    def __init__(self, ndim, nhidden=256, nlayers=5, time_embed_dim=64,
                 sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.ndim = ndim
        self.sigma_min = sigma_min   # needed by edm_sampler
        self.sigma_max = sigma_max
        self.backbone = UNetBlockStyleMLP_backbone(
            ndim=ndim,
            nlayers=nlayers,
            nhidden=nhidden,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, x, sigma, cond=None):
        """
        Args:
            x:     noisy input  [B, ndim]
            sigma: noise level  [B] or scalar
        Returns:
            denoised prediction [B, ndim]
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
        sigma = sigma.to(x.device)
        # Input scaling: whiten by noise level
        sigma_vec = sigma.view(-1, 1)
        x_in = x / (1.0 + sigma_vec ** 2).sqrt()
        # Noise embedding: log(sigma)/4, same as EDM convention
        t_enc = sigma.log() / 4.0
        return self.backbone(x_in, t_enc)
