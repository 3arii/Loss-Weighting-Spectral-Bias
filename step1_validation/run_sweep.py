"""Main training script: shared-W linear denoiser with power-law loss weighting.

For each (beta, alpha_data, ndim, seed) configuration:
1. Generate Gaussian data with power-law eigenvalue spectrum
2. Compute all theory predictions (heuristic, Phi integral, shared-W ODE)
3. Train LinearDenoiserShared with SGD + normalized loss
4. Checkpoint W diagonal at geometric steps
5. Compute emergence times (both sampling-based and a_k-based), fit power law
6. Save JSON with full results

Usage:
    python -m step1_validation.run_sweep --beta -1.0 --alpha_data 1.0 --ndim 200 --seed 42
    python -m step1_validation.run_sweep --beta 0.0 --alpha_data 1.0 --ndim 10 --seed 42 --max_steps 5000
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import (
    SIGMA_0, SIGMA_T, K_SIGMA, LR, MAX_STEPS, N_SAMPLES, GRAD_CLIP_NORM,
    N_CHECKPOINTS, ETA, Q_K, THRESHOLD_FRAC,
    get_sigma_grid, get_sigma_grid_np, get_checkpoint_steps, make_eigenvalues,
)
from .models import LinearDenoiserShared
from .losses import DeterministicPowerLawLoss
from .theory import (
    compute_phi_per_sigma, compute_A_k, compute_shared_w_trajectory,
    compute_shared_w_variance, compute_inaccessible_modes,
    compute_emergence_times, compute_emergence_times_ak,
    compute_emergence_times_ak_analytical, fit_power_law,
)


def _fmt(val, fmt=".3f"):
    """Format float or NaN safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "nan"
    return f"{val:{fmt}}"


def generate_data(alpha_data, ndim, n_samples, seed):
    """Generate X ~ N(0, diag(k^{-alpha_data})) with given seed."""
    rng = np.random.default_rng(seed)
    eigenvalues = make_eigenvalues(alpha_data, ndim)
    std = np.sqrt(eigenvalues)
    X = rng.standard_normal((n_samples, ndim)) * std[None, :]
    return torch.tensor(X, dtype=torch.float32), eigenvalues


def compute_theory_predictions(eigenvalues, beta, sigma_grid_np, w_values_np):
    """Compute all analytical predictions for this configuration."""
    d = len(eigenvalues)

    # 1. Heuristic
    alpha_heuristic = 1.0 + beta / 2.0

    # 2. Per-sigma Phi integral (fewer tau points for speed; 500 suffices)
    tau_array = np.geomspace(1e-2, 1e6, 500)

    def w_fn(sigma):
        return sigma ** beta

    variance_phi = compute_phi_per_sigma(
        tau_array, eigenvalues, q_k=Q_K, eta=ETA, w_fn=w_fn,
        sigma_0=SIGMA_0, sigma_T=SIGMA_T, n_quad=100
    )
    tau_phi = compute_emergence_times(variance_phi, tau_array, eigenvalues,
                                      threshold_frac=THRESHOLD_FRAC)
    fit_phi = fit_power_law(tau_phi, eigenvalues)

    # 3. Shared-W analytical theory
    A_k, a_k_star, sigma_eff_sq = compute_A_k(eigenvalues, w_values_np, sigma_grid_np)

    # Normalize A_k consistently with the loss normalization
    A_max_theory = (w_values_np * (eigenvalues[0] + sigma_grid_np**2)).mean()
    A_k_norm = A_k / A_max_theory
    a_k_star_norm = eigenvalues * (w_values_np.mean() / A_max_theory) / A_k_norm

    # Shared-W sampling-based emergence (may be all-inaccessible)
    sw_traj = compute_shared_w_trajectory(tau_array, eigenvalues, A_k_norm, a_k_star_norm)
    variance_sw = compute_shared_w_variance(sw_traj)
    tau_sw_sampling = compute_emergence_times(variance_sw, tau_array, eigenvalues,
                                              threshold_frac=THRESHOLD_FRAC)
    fit_sw_sampling = fit_power_law(tau_sw_sampling, eigenvalues)

    # Shared-W convergence-rate emergence (a_k reaching 90% of a_k*)
    tau_sw_ak = compute_emergence_times_ak(sw_traj, tau_array, a_k_star_norm,
                                           convergence_frac=0.9)
    fit_sw_ak = fit_power_law(tau_sw_ak, eigenvalues)

    # Analytical convergence-rate emergence
    tau_sw_ak_analytical = compute_emergence_times_ak_analytical(A_k_norm, convergence_frac=0.9)
    fit_sw_ak_analytical = fit_power_law(tau_sw_ak_analytical, eigenvalues)

    # Inaccessible modes
    inaccessible_mask, lambda_tilde_inf = compute_inaccessible_modes(
        eigenvalues, a_k_star_norm, threshold_frac=THRESHOLD_FRAC
    )

    return {
        "alpha_heuristic": alpha_heuristic,
        # Per-sigma Phi integral
        "alpha_phi_per_sigma": fit_phi["alpha"],
        "alpha_phi_R2": fit_phi["R2"],
        "alpha_phi_n_used": fit_phi["n_used"],
        # Shared-W sampling-based (may be NaN if all inaccessible)
        "alpha_shared_W_theory_sampling": fit_sw_sampling["alpha"],
        "alpha_shared_W_theory_sampling_R2": fit_sw_sampling["R2"],
        # Shared-W convergence-rate (a_k-based)
        "alpha_shared_W_theory_ak": fit_sw_ak["alpha"],
        "alpha_shared_W_theory_ak_R2": fit_sw_ak["R2"],
        # Shared-W convergence-rate (analytical, no trajectory needed)
        "alpha_shared_W_theory_ak_analytical": fit_sw_ak_analytical["alpha"],
        "alpha_shared_W_theory_ak_analytical_R2": fit_sw_ak_analytical["R2"],
        # Metadata
        "sigma_eff_squared": float(sigma_eff_sq),
        "n_modes_inaccessible": int(np.sum(inaccessible_mask)),
        "inaccessible_mask": inaccessible_mask.tolist(),
        "A_k_normalized": A_k_norm.tolist(),
        "a_k_star_normalized": a_k_star_norm.tolist(),
    }


def train_and_measure(X, eigenvalues, beta, ndim, lr, max_steps, device, a_k_star_norm):
    """Train shared-W model and measure emergence times."""
    model = LinearDenoiserShared(ndim).to(device)
    loss_fn = DeterministicPowerLawLoss(beta, lambda_max=float(eigenvalues[0]))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X_device = X.to(device)

    checkpoint_steps = get_checkpoint_steps(max_steps, N_CHECKPOINTS)
    checkpoint_set = set(checkpoint_steps.tolist())

    a_k_trajectories = []
    checkpoint_step_list = []
    loss_trajectory = []
    max_grad_norm = 0.0

    t0 = time.time()
    for step in range(max_steps):
        loss = loss_fn(model, X_device)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        max_grad_norm = max(max_grad_norm, float(grad_norm))
        optimizer.step()

        if step % 10 == 0:
            loss_trajectory.append(float(loss.item()))

        if step in checkpoint_set:
            a_k = model.W.data.diag().cpu().numpy().copy()
            a_k_trajectories.append(a_k)
            checkpoint_step_list.append(step)

    train_time = time.time() - t0

    a_k_trajectories = np.array(a_k_trajectories)  # [n_ckpt, d]
    checkpoint_steps_arr = np.array(checkpoint_step_list, dtype=np.float64)

    # Sampling-based emergence (may be all NaN)
    variance_traj = compute_shared_w_variance(a_k_trajectories)
    tau_sampling = compute_emergence_times(
        variance_traj, checkpoint_steps_arr, eigenvalues, threshold_frac=THRESHOLD_FRAC
    )
    fit_sampling = fit_power_law(tau_sampling, eigenvalues)

    # Convergence-rate emergence (a_k reaching 90% of a_k*)
    tau_ak = compute_emergence_times_ak(
        a_k_trajectories, checkpoint_steps_arr, a_k_star_norm, convergence_frac=0.9
    )
    fit_ak = fit_power_law(tau_ak, eigenvalues)

    n_emerged_sampling = int(np.sum(np.isfinite(tau_sampling)))
    n_emerged_ak = int(np.sum(np.isfinite(tau_ak)))

    return {
        "a_k_trajectories": a_k_trajectories.tolist(),
        "variance_trajectories": variance_traj.tolist(),
        "checkpoint_steps": checkpoint_step_list,
        # Sampling-based
        "emergence_times_sampling": np.where(np.isnan(tau_sampling), None, tau_sampling).tolist(),
        "alpha_trained_sampling": fit_sampling["alpha"],
        "alpha_trained_sampling_R2": fit_sampling["R2"],
        "n_emerged_sampling": n_emerged_sampling,
        # Convergence-rate (a_k-based)
        "emergence_times_ak": np.where(np.isnan(tau_ak), None, tau_ak).tolist(),
        "alpha_trained_ak": fit_ak["alpha"],
        "alpha_trained_ak_R2": fit_ak["R2"],
        "n_emerged_ak": n_emerged_ak,
        # Training metadata
        "loss_trajectory": loss_trajectory,
        "max_gradient_norm": max_grad_norm,
        "train_time_seconds": train_time,
        "normalization_factor": loss_fn.normalization_factor,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 1: Shared-W sweep")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--alpha_data", type=float, required=True)
    parser.add_argument("--ndim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Config: beta={args.beta}, alpha_data={args.alpha_data}, "
          f"d={args.ndim}, seed={args.seed}, device={args.device}")

    # Generate data
    X, eigenvalues = generate_data(args.alpha_data, args.ndim, args.n_samples, args.seed)

    # Sigma grid for theory
    sigma_grid_np = get_sigma_grid_np()
    w_values_np = sigma_grid_np ** args.beta

    # Theory predictions
    print("Computing theory predictions...")
    theory = compute_theory_predictions(eigenvalues, args.beta, sigma_grid_np, w_values_np)

    print(f"  alpha_heuristic     = {_fmt(theory['alpha_heuristic'])}")
    print(f"  alpha_phi           = {_fmt(theory['alpha_phi_per_sigma'])} "
          f"(R2={_fmt(theory['alpha_phi_R2'], '.4f')})")
    print(f"  alpha_SW_ak_theory  = {_fmt(theory['alpha_shared_W_theory_ak'])} "
          f"(R2={_fmt(theory['alpha_shared_W_theory_ak_R2'], '.4f')})")
    print(f"  sigma_eff^2         = {theory['sigma_eff_squared']:.1f}")
    print(f"  inaccessible        = {theory['n_modes_inaccessible']}/{args.ndim} "
          f"(sampling ODE metric)")

    # Train
    a_k_star_norm = np.array(theory["a_k_star_normalized"])
    print(f"Training ({args.max_steps} steps, lr={args.lr})...")
    training = train_and_measure(
        X, eigenvalues, args.beta, args.ndim, args.lr, args.max_steps, args.device,
        a_k_star_norm
    )

    print(f"  alpha_trained_ak    = {_fmt(training['alpha_trained_ak'])} "
          f"(R2={_fmt(training['alpha_trained_ak_R2'], '.4f')}, "
          f"emerged={training['n_emerged_ak']}/{args.ndim})")
    print(f"  alpha_trained_samp  = {_fmt(training['alpha_trained_sampling'])} "
          f"(emerged={training['n_emerged_sampling']}/{args.ndim})")
    print(f"  max_grad_norm       = {training['max_gradient_norm']:.2f}")
    print(f"  train_time          = {training['train_time_seconds']:.1f}s")

    # Assemble output
    result = {
        "config": {
            "beta": args.beta,
            "alpha_data": args.alpha_data,
            "ndim": args.ndim,
            "seed": args.seed,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "n_samples": args.n_samples,
            "K_sigma": K_SIGMA,
            "sigma_0": SIGMA_0,
            "sigma_T": SIGMA_T,
            "threshold_frac": THRESHOLD_FRAC,
        },
        "eigenvalues": eigenvalues.tolist(),
        **theory,
        **training,
    }

    # Save
    fname = (f"beta_{args.beta:.1f}_alpha_{args.alpha_data:.2f}"
             f"_d_{args.ndim}_seed_{args.seed}.json")
    outpath = os.path.join(args.output_dir, fname)
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
