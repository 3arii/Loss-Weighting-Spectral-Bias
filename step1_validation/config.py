"""Constants and sweep parameters for Step 1 power-law validation."""

import numpy as np
import torch

# Noise schedule bounds
SIGMA_0 = 0.002
SIGMA_T = 80.0
K_SIGMA = 50

# Training
LR = 0.01
MAX_STEPS = 50000
N_SAMPLES = 10000
GRAD_CLIP_NORM = 100.0

# Emergence threshold
THRESHOLD_FRAC = 0.5

# Checkpointing
N_CHECKPOINTS = 200

# Theory
ETA = 1.0
Q_K = 0.0

# Sweep grid — power-law only
BETA_VALUES = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ALPHA_DATA_VALUES = [0.56, 1.0, 2.0]
SEEDS = [42, 43, 44, 45, 46]


def get_sigma_grid_np(K=K_SIGMA, sigma_min=SIGMA_0, sigma_max=SIGMA_T):
    return np.logspace(np.log10(sigma_min), np.log10(sigma_max), K)


def get_checkpoint_steps(max_steps=MAX_STEPS, n_checkpoints=N_CHECKPOINTS):
    steps = np.unique(np.geomspace(1, max_steps, n_checkpoints).astype(int))
    steps = np.concatenate([[0], steps])
    return np.unique(np.clip(steps, 0, max_steps - 1))


def make_eigenvalues(alpha_data, d):
    return np.arange(1, d + 1, dtype=np.float64) ** (-alpha_data)
