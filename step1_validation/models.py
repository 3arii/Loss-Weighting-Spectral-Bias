"""Denoiser models for Step 1 validation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# MLP denoiser — ported from DiffusionLearningCurve/core/diffusion_nn_lib.py
# (UNetBlockStyleMLP_backbone_NoFirstNorm at line 231 — the "pure" MLP Binxu
# recommended as the fail-fast starting point: no EDM preconditioning,
# no output skip connection, just a stack of adaptive-norm MLP blocks with
# a Gaussian-Fourier sigma embedding).
# ---------------------------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """Sigma embedding: fixed random Fourier features."""

    def __init__(self, embed_dim, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale,
                              requires_grad=False)

    def forward(self, t):
        t_proj = t.view(-1, 1) * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class UNetMLPBlock(nn.Module):
    """Adaptive-norm residual MLP block (EDM-style AdaGN scale+shift)."""

    def __init__(self, in_features, out_features, emb_features,
                 dropout=0.0, skip_scale=1.0, eps=1e-5, adaptive_scale=True):
        super().__init__()
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.dropout = dropout

        self.norm0 = nn.LayerNorm(in_features, eps=eps)
        self.fc0 = nn.Linear(in_features, out_features)
        self.affine = nn.Linear(emb_features,
                                out_features * (2 if adaptive_scale else 1))
        self.norm1 = nn.LayerNorm(out_features, eps=eps)
        self.fc1 = nn.Linear(out_features, out_features)

        self.skip = (nn.Linear(in_features, out_features)
                     if in_features != out_features else None)

    def forward(self, x, emb):
        orig = x
        x = self.fc0(F.silu(self.norm0(x)))
        params = self.affine(emb).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x + params))
        x = self.fc1(F.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        return x * self.skip_scale


class UNetBlockStyleMLP_backbone_NoFirstNorm(nn.Module):
    """Pure MLP backbone. Expects the sigma-encoding to be passed in as `t_enc`
    (e.g. log(sigma)/4). A pre-projection layer lifts low-dim inputs into the
    hidden width before the first LayerNorm."""

    def __init__(self, ndim, nlayers=5, nhidden=64, time_embed_dim=64):
        super().__init__()
        self.embed = GaussianFourierProjection(time_embed_dim, scale=1.0)
        layers = nn.ModuleList()
        if ndim < nhidden:
            layers.append(nn.Linear(ndim, nhidden))
            first_in = nhidden
        else:
            first_in = ndim
        layers.append(UNetMLPBlock(first_in, nhidden, time_embed_dim))
        for _ in range(nlayers - 2):
            layers.append(UNetMLPBlock(nhidden, nhidden, time_embed_dim))
        layers.append(nn.Linear(nhidden, ndim))
        self.net = layers

    def forward(self, x, t_enc, cond=None):
        t_embed = self.embed(t_enc)
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = layer(x, t_embed)
        return self.net[-1](x)


class MLPDenoiser(nn.Module):
    """Pure MLP denoiser: D(x, sigma) -> predicted clean x.

    No EDM input scaling, no output skip connection. Sigma is fed to the
    backbone as log(sigma)/4, which is just a representation choice
    (sigma lives on a log axis), not EDM-style preconditioning.
    """

    def __init__(self, ndim, nlayers=5, nhidden=64, time_embed_dim=64):
        super().__init__()
        self.backbone = UNetBlockStyleMLP_backbone_NoFirstNorm(
            ndim=ndim, nlayers=nlayers, nhidden=nhidden,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, x, sigma, cond=None):
        if sigma.ndim == 0:
            sigma = sigma.expand(x.shape[0])
        else:
            sigma = sigma.view(-1)
        t_enc = torch.log(sigma) / 4.0
        return self.backbone(x, t_enc, cond=cond)
