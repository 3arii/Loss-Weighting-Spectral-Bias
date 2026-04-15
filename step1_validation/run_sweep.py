"""Train shared-W linear denoiser with power-law loss weighting.

Usage:
    python -m step1_validation.run_sweep --beta -1.0 --alpha_data 1.0 --ndim 200 --seed 42 --output_dir results
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
    get_sigma_grid_np, get_checkpoint_steps, make_eigenvalues,
)
from .models import LinearDenoiserShared
from .losses import DeterministicPowerLawLoss
from .theory import (
    compute_phi_per_sigma, compute_A_k, compute_shared_w_variance,
    compute_emergence_times, compute_emergence_times_ak, fit_power_law,
)


def _fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "nan"
    return f"{val:.3f}"


def generate_data(alpha_data, ndim, n_samples, seed):
    rng = np.random.default_rng(seed)
    eigenvalues = make_eigenvalues(alpha_data, ndim)
    X = rng.standard_normal((n_samples, ndim)) * np.sqrt(eigenvalues)[None, :]
    return torch.tensor(X, dtype=torch.float32), eigenvalues


def compute_theory(eigenvalues, beta, sigma_grid_np, w_values_np):
    """All theory predictions for one config."""
    d = len(eigenvalues)

    # 1. Heuristic
    alpha_heuristic = 1.0 + beta / 2.0

    # 2. Per-sigma Phi integral
    tau_theory = np.geomspace(1e-2, 1e6, 500)
    w_fn = lambda sigma: sigma ** beta
    var_phi = compute_phi_per_sigma(tau_theory, eigenvalues, q_k=Q_K, eta=ETA,
                                    w_fn=w_fn, n_quad=100)
    tau_phi = compute_emergence_times(var_phi, tau_theory, eigenvalues)
    fit_phi = fit_power_law(tau_phi, eigenvalues)

    # 3. Shared-W theory (a_k-based emergence)
    A_k, a_k_star, sigma_eff_sq = compute_A_k(eigenvalues, w_values_np, sigma_grid_np)
    A_max = (w_values_np * (eigenvalues[0] + sigma_grid_np**2)).mean()
    A_k_norm = A_k / A_max
    a_k_star_norm = eigenvalues * (w_values_np.mean() / A_max) / A_k_norm

    # Analytical a_k trajectory
    a_k_traj = a_k_star_norm[None, :] * (1.0 - np.exp(-2.0 * ETA * tau_theory[:, None] * A_k_norm[None, :]))
    tau_ak = compute_emergence_times_ak(a_k_traj, tau_theory, a_k_star_norm, 0.9)
    fit_ak = fit_power_law(tau_ak, eigenvalues)

    # Inaccessible modes (sampling ODE)
    var_inf = compute_shared_w_variance(a_k_star_norm)
    n_inaccessible = int(np.sum(var_inf < THRESHOLD_FRAC * eigenvalues))

    return {
        "alpha_heuristic": alpha_heuristic,
        "alpha_phi": fit_phi["alpha"],
        "alpha_phi_R2": fit_phi["R2"],
        "alpha_phi_n_used": fit_phi["n_used"],
        "alpha_shared_W_ak_theory": fit_ak["alpha"],
        "alpha_shared_W_ak_theory_R2": fit_ak["R2"],
        "sigma_eff_squared": float(sigma_eff_sq),
        "n_inaccessible": n_inaccessible,
        "A_k_norm": A_k_norm.tolist(),
        "a_k_star_norm": a_k_star_norm.tolist(),
    }


def train(X, eigenvalues, beta, ndim, lr, max_steps, device, a_k_star_norm):
    """Train and measure emergence."""
    model = LinearDenoiserShared(ndim).to(device)
    loss_fn = DeterministicPowerLawLoss(beta, lambda_max=float(eigenvalues[0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X_dev = X.to(device)

    ckpt_steps = get_checkpoint_steps(max_steps, N_CHECKPOINTS)
    ckpt_set = set(ckpt_steps.tolist())

    a_k_list, ckpt_list, loss_list = [], [], []
    max_grad = 0.0

    t0 = time.time()
    for step in range(max_steps):
        loss = loss_fn(model, X_dev)
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        max_grad = max(max_grad, float(gn))
        optimizer.step()

        if step % 50 == 0:
            loss_list.append(float(loss.item()))
        if step in ckpt_set:
            a_k_list.append(model.W.data.diag().cpu().numpy().copy())
            ckpt_list.append(step)

    train_time = time.time() - t0
    a_k_traj = np.array(a_k_list)
    ckpt_arr = np.array(ckpt_list, dtype=np.float64)

    # Sampling-based emergence
    var_traj = compute_shared_w_variance(a_k_traj)
    tau_samp = compute_emergence_times(var_traj, ckpt_arr, eigenvalues)
    fit_samp = fit_power_law(tau_samp, eigenvalues)

    # a_k-based emergence
    tau_ak = compute_emergence_times_ak(a_k_traj, ckpt_arr,
                                        np.array(a_k_star_norm), 0.9)
    fit_ak = fit_power_law(tau_ak, eigenvalues)

    return {
        "a_k_traj": a_k_traj.tolist(),
        "var_traj": var_traj.tolist(),
        "ckpt_steps": ckpt_list,
        "alpha_trained_sampling": fit_samp["alpha"],
        "alpha_trained_sampling_R2": fit_samp["R2"],
        "n_emerged_sampling": int(np.sum(np.isfinite(tau_samp))),
        "alpha_trained_ak": fit_ak["alpha"],
        "alpha_trained_ak_R2": fit_ak["R2"],
        "n_emerged_ak": int(np.sum(np.isfinite(tau_ak))),
        "loss_traj": loss_list,
        "max_grad_norm": max_grad,
        "train_time_s": train_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--alpha_data", type=float, required=True)
    p.add_argument("--ndim", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--n_samples", type=int, default=N_SAMPLES)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"beta={args.beta} alpha_data={args.alpha_data} d={args.ndim} "
          f"seed={args.seed} device={args.device}")

    X, eigenvalues = generate_data(args.alpha_data, args.ndim, args.n_samples, args.seed)
    sigma_grid = get_sigma_grid_np()
    w_values = sigma_grid ** args.beta

    print("Theory...")
    theory = compute_theory(eigenvalues, args.beta, sigma_grid, w_values)
    print(f"  heuristic={_fmt(theory['alpha_heuristic'])}  "
          f"phi={_fmt(theory['alpha_phi'])} (R2={_fmt(theory['alpha_phi_R2'])})  "
          f"inacc={theory['n_inaccessible']}/{args.ndim}")

    print(f"Training {args.max_steps} steps...")
    trained = train(X, eigenvalues, args.beta, args.ndim, args.lr,
                    args.max_steps, args.device, theory["a_k_star_norm"])
    print(f"  ak={_fmt(trained['alpha_trained_ak'])} "
          f"(R2={_fmt(trained['alpha_trained_ak_R2'])}, "
          f"emerged={trained['n_emerged_ak']}/{args.ndim})  "
          f"time={trained['train_time_s']:.0f}s")

    result = {
        "config": vars(args),
        "eigenvalues": eigenvalues.tolist(),
        **theory,
        **trained,
    }

    fname = f"b{args.beta:+.1f}_a{args.alpha_data:.2f}_d{args.ndim}_s{args.seed}.json"
    path = os.path.join(args.output_dir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
