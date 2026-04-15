"""Centralized constants and sweep parameters for Step 1 validation."""

import numpy as np
import torch

# Noise schedule bounds
SIGMA_0 = 0.002
SIGMA_T = 80.0
K_SIGMA = 50  # number of sigma grid points

# Training
LR = 0.01
MAX_STEPS = 50000
N_SAMPLES = 10000
GRAD_CLIP_NORM = 100.0

# Emergence threshold
THRESHOLD_FRAC = 0.5  # lambda_tilde_k >= 0.5 * lambda_k

# Checkpointing
N_CHECKPOINTS = 200

# Theory
ETA = 1.0        # gradient flow learning rate (theory)
Q_K = 0.0        # initial overlap (zero init: W=0 => a_k=0)
SIGMA_DATA = 0.5  # EDM convention

# Sweep grid
# β=2.5 dropped: only 1/200 modes emerges within τ_max (see DIAGNOSIS_beta2p5_nan.md)
BETA_VALUES = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ALPHA_DATA_VALUES = [0.56, 1.0, 2.0]
D_VALUES = [200, 768]
SEEDS = [42, 43, 44, 45, 46]

# Named weighting schemes for comparison
NAMED_SCHEMES = ["uniform", "edm", "min-snr-gamma", "p2"]


def get_sigma_grid(K=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T):
    """Log-spaced sigma grid as a torch tensor."""
    return torch.logspace(np.log10(sigma_min), np.log10(sigma_max), K)


def get_sigma_grid_np(K=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T):
    """Log-spaced sigma grid as a numpy array."""
    return np.logspace(np.log10(sigma_min), np.log10(sigma_max), K)


def get_checkpoint_steps(max_steps=MAX_STEPS, n_checkpoints=N_CHECKPOINTS):
    """Geometrically-spaced checkpoint steps, always including step 0 and max_steps-1."""
    steps = np.unique(np.geomspace(1, max_steps, n_checkpoints).astype(int))
    steps = np.concatenate([[0], steps])
    steps = np.unique(np.clip(steps, 0, max_steps - 1))
    return steps


def make_eigenvalues(alpha_data, d):
    """Power-law eigenvalue spectrum: lambda_k = k^{-alpha_data}, normalized so lambda_1 = 1."""
    k = np.arange(1, d + 1, dtype=np.float64)
    return k ** (-alpha_data)
