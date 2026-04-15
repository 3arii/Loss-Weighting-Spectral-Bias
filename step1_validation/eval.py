"""Sampling-based evaluation for Step 1 validation.

Pipeline (mirrors DiffusionLearningCurve posthoc_spectral_convergence_analysis):
  1. edm_sampler: generate samples from model via Heun ODE
  2. compute_data_eigenbasis: PCA of training data -> (eigval, eigvec)
  3. compute_sample_covariance_in_eigenbasis: project generated covariance to data eigenbasis
     -> variance per eigenmode lambda_tilde_k
  4. run_covariance_trajectory: do (1)+(3) at every checkpoint -> [n_ckpts, d]
  5. compute_emergence_steps: find when lambda_tilde_k crosses threshold * lambda_k
  6. fit_alpha: power law fit emergence_step ~ lambda_k^{-alpha}
"""

import sys
import numpy as np
import torch

_DLC_PATH = "/n/home12/binxuwang/Github/DiffusionLearningCurve"
if _DLC_PATH not in sys.path:
    sys.path.insert(0, _DLC_PATH)

from core.diffusion_edm_lib import edm_sampler           # noqa: E402
from core.trajectory_convergence_lib import (            # noqa: E402
    compute_crossing_points,
    fit_regression_log_scale,
)

from .config import SIGMA_0, SIGMA_T, THRESHOLD_FRAC


# ---------------------------------------------------------------------------
# 1. Eigenbasis of training data
# ---------------------------------------------------------------------------

def compute_data_eigenbasis(X: torch.Tensor, device="cpu"):
    """PCA of training data X [N, d] -> (eigval [d], eigvec [d, d]).

    eigval sorted descending. eigvec columns are eigenvectors.
    """
    X = X.to(device).float()
    X_c = X - X.mean(0)
    cov = X_c.T @ X_c / (X.shape[0] - 1)          # [d, d]
    eigval, eigvec = torch.linalg.eigh(cov)         # ascending
    # Sort descending
    idx     = eigval.argsort(descending=True)
    eigval  = eigval[idx]
    eigvec  = eigvec[:, idx]                        # columns = eigenvectors
    return eigval.cpu(), eigvec.cpu()


# ---------------------------------------------------------------------------
# 2. Sample covariance projected to data eigenbasis
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, n_samples: int, ndim: int, device="cpu",
                     num_steps=50, sigma_min=SIGMA_0, sigma_max=SIGMA_T):
    """Generate n_samples from model using EDM Heun ODE sampler.

    Model interface: D(x_noisy [B,d], sigma [B]) -> x_denoised [B,d]
    Returns: [n_samples, ndim] on CPU.
    """
    model.eval()
    latents = torch.randn(n_samples, ndim, device=device)
    x_gen   = edm_sampler(
        model, latents,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        return_traj=False,
    )
    return x_gen.cpu()


def compute_sample_covariance_in_eigenbasis(x_gen: torch.Tensor,
                                            eigvec: torch.Tensor) -> torch.Tensor:
    """Project generated sample covariance into data eigenbasis.

    Args:
        x_gen:   generated samples [N, d]
        eigvec:  data eigenvectors [d, d] (columns = eigenvectors)
    Returns:
        var_k:   variance per eigenmode [d]  (diagonal of U^T Sigma_gen U)
    """
    x_gen = x_gen.float()
    cov   = torch.cov(x_gen.T)                     # [d, d]
    cov_proj = eigvec.T @ cov @ eigvec              # [d, d]
    return cov_proj.diag()                          # [d]


# ---------------------------------------------------------------------------
# 3. Full covariance trajectory across checkpoints
# ---------------------------------------------------------------------------

def run_covariance_trajectory(model_state_dicts: list,
                               model_factory,
                               checkpoint_steps: list,
                               X_train: torch.Tensor,
                               ndim: int,
                               n_eval_samples: int = 2000,
                               num_ode_steps: int = 50,
                               device: str = "cpu") -> dict:
    """Compute per-eigenmode variance at each checkpoint.

    Args:
        model_state_dicts:  list of state_dicts from saved checkpoints
        model_factory:      callable () -> nn.Module (fresh model)
        checkpoint_steps:   list of training step indices (same length)
        X_train:            training data [N, d] for eigenbasis
        ndim:               data dimensionality
        n_eval_samples:     samples to generate per checkpoint
        num_ode_steps:      ODE steps for sampler
        device:             "cuda" or "cpu"
    Returns:
        dict with keys:
            eigval            [d]           true data eigenvalues
            eigvec            [d, d]        data eigenvectors
            var_traj          [n_ckpts, d]  generated variance per mode per ckpt
            checkpoint_steps  list
    """
    eigval, eigvec = compute_data_eigenbasis(X_train, device=device)
    eigvec_cpu = eigvec.cpu()

    var_list = []
    for state_dict in model_state_dicts:
        model = model_factory().to(device)
        model.load_state_dict(state_dict)
        x_gen = generate_samples(model, n_eval_samples, ndim, device=device,
                                  num_steps=num_ode_steps)
        var_k = compute_sample_covariance_in_eigenbasis(x_gen, eigvec_cpu)
        var_list.append(var_k.numpy())

    var_traj = np.stack(var_list, axis=0)           # [n_ckpts, d]

    return {
        "eigval":           eigval.numpy(),
        "eigvec":           eigvec.numpy(),
        "var_traj":         var_traj,
        "checkpoint_steps": checkpoint_steps,
    }


# ---------------------------------------------------------------------------
# 4. Emergence detection + power-law fit
# ---------------------------------------------------------------------------

def compute_emergence_and_fit(eigval: np.ndarray,
                               var_traj: np.ndarray,
                               checkpoint_steps: list,
                               threshold_frac: float = THRESHOLD_FRAC,
                               threshold_type: str = "harmonic_mean") -> dict:
    """Find emergence steps and fit alpha: emergence_step ~ lambda_k^{-alpha}.

    Args:
        eigval:           true eigenvalues [d]
        var_traj:         generated variance trajectory [n_ckpts, d]
        checkpoint_steps: training steps [n_ckpts]
        threshold_frac:   fraction for range-threshold mode
        threshold_type:   "harmonic_mean" | "mean" | "geometric_mean" | "range"
    Returns:
        dict with:
            df               DataFrame (Variance, emergence_step, Direction)
            alpha            fitted power-law exponent
            R2               R² of fit
            n_emerged        number of modes that emerged
    """
    step_arr = np.array(checkpoint_steps, dtype=float)

    df = compute_crossing_points(
        target_eigval    = torch.as_tensor(eigval),
        empiric_var_traj = var_traj,
        step_slice       = step_arr,
        threshold_type   = threshold_type,
        smooth_sigma     = 1.0,
        threshold_fraction = threshold_frac,
    )

    # Keep only modes that emerged (finite emergence_step)
    df_valid = df[df["emergence_step"].notna()].copy()
    n_emerged = len(df_valid)

    if n_emerged < 2:
        return {"df": df, "alpha": float("nan"), "R2": float("nan"),
                "n_emerged": n_emerged}

    fit = fit_regression_log_scale(
        df_valid["Variance"].values,
        df_valid["emergence_step"].values,
    )
    # emergence_step ~ lambda^b  =>  alpha = -b  (larger lambda emerges faster = smaller step)
    alpha = -fit["slope"]

    return {
        "df":        df,
        "alpha":     alpha,
        "R2":        fit["r_squared"],
        "n_emerged": n_emerged,
        "fit":       fit,
    }
