"""Train per-sigma linear denoiser with power-law loss weighting.

Each sigma level has its own independent diagonal denoiser.
This matches the theory setup exactly, so trained alpha should match Phi integral.

Usage:
    python -m step1_validation.run_sweep --beta 0.0 --alpha_data 1.0 --ndim 200 --seed 42
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
from .models import LinearDenoiserPerSigma
from .losses import PerSigmaPowerLawLoss
from .theory import (
    compute_phi_per_sigma, compute_emergence_times, fit_power_law,
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


def compute_theory(eigenvalues, beta):
    """Phi integral predictions (no training needed)."""
    alpha_heuristic = 1.0 + beta / 2.0

    tau_theory = np.geomspace(1e-2, 1e6, 500)
    w_fn = lambda sigma: sigma ** beta
    var_phi = compute_phi_per_sigma(tau_theory, eigenvalues, q_k=Q_K, eta=ETA,
                                    w_fn=w_fn, n_quad=100)
    tau_phi = compute_emergence_times(var_phi, tau_theory, eigenvalues)
    fit_phi = fit_power_law(tau_phi, eigenvalues)

    return {
        "alpha_heuristic": alpha_heuristic,
        "alpha_phi": fit_phi["alpha"],
        "alpha_phi_R2": fit_phi["R2"],
        "alpha_phi_n_used": fit_phi["n_used"],
        "emergence_times_phi": np.where(np.isnan(tau_phi), None, tau_phi).tolist(),
    }


def compute_generated_variance_per_sigma(model, sigma_grid, sigma_0, sigma_T):
    """Compute lambda_tilde_k from per-sigma a_k values using the Phi integral formula.

    For per-sigma model with constant a_k(sigma_j) at each sigma:
        Phi_k(sigma) = exp(-integral (psi_k - 1)/s ds)

    Since psi_k(sigma_j) = a_k[j] (the learned diagonal), and psi varies with sigma,
    we numerically integrate using the a_k values at the sigma grid points.

    lambda_tilde_k = sigma_T^2 * (Phi_k(sigma_0) / Phi_k(sigma_T))^2
    """
    # a_k: [K_sigma, d] from model parameters
    a_k = model.a_k.data.cpu().numpy()  # [K, d]
    K = len(sigma_grid)

    # Integrate (a_k - 1) in log-sigma space using trapezoidal rule
    # The integrand at each sigma_j is (a_k[j] - 1), integrated in log(sigma)
    log_sigma = np.log(sigma_grid)

    integrand = a_k - 1.0  # [K, d]

    # Trapezoidal integration over log-sigma from sigma_0 to sigma_T
    # This gives log(Phi_0/Phi_T) = integral_{ln(sigma_0)}^{ln(sigma_T)} (psi-1) d(ln sigma)
    log_ratio = np.trapz(integrand, log_sigma, axis=0)  # [d]

    lambda_tilde = sigma_T**2 * np.exp(2.0 * log_ratio)
    return lambda_tilde


def train(X, eigenvalues, beta, ndim, lr, max_steps, device):
    """Train per-sigma model and measure emergence via generated variance."""
    K = K_SIGMA
    model = LinearDenoiserPerSigma(ndim, K).to(device)
    loss_fn = PerSigmaPowerLawLoss(beta, lambda_max=float(eigenvalues[0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X_dev = X.to(device)

    sigma_grid = get_sigma_grid_np()
    ckpt_steps = get_checkpoint_steps(max_steps, N_CHECKPOINTS)
    ckpt_set = set(ckpt_steps.tolist())

    var_list, ckpt_list, loss_list = [], [], []
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
            lam_tilde = compute_generated_variance_per_sigma(
                model, sigma_grid, SIGMA_0, SIGMA_T)
            var_list.append(lam_tilde)
            ckpt_list.append(step)

    train_time = time.time() - t0
    var_traj = np.array(var_list)  # [n_ckpt, d]
    ckpt_arr = np.array(ckpt_list, dtype=np.float64)

    # Emergence times from generated variance
    tau_trained = compute_emergence_times(var_traj, ckpt_arr, eigenvalues)
    fit_trained = fit_power_law(tau_trained, eigenvalues)

    return {
        "var_traj": var_traj.tolist(),
        "ckpt_steps": ckpt_list,
        "emergence_times_trained": np.where(np.isnan(tau_trained), None, tau_trained).tolist(),
        "alpha_trained": fit_trained["alpha"],
        "alpha_trained_R2": fit_trained["R2"],
        "alpha_trained_n_used": fit_trained["n_used"],
        "n_emerged": int(np.sum(np.isfinite(tau_trained))),
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

    print("Theory...")
    theory = compute_theory(eigenvalues, args.beta)
    print(f"  heuristic={_fmt(theory['alpha_heuristic'])}  "
          f"phi={_fmt(theory['alpha_phi'])} (R2={_fmt(theory['alpha_phi_R2'])})")

    print(f"Training {args.max_steps} steps (per-sigma model, K={K_SIGMA})...")
    trained = train(X, eigenvalues, args.beta, args.ndim, args.lr,
                    args.max_steps, args.device)
    print(f"  alpha_trained={_fmt(trained['alpha_trained'])} "
          f"(R2={_fmt(trained['alpha_trained_R2'])}, "
          f"emerged={trained['n_emerged']}/{args.ndim})  "
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
