"""Train shared-MLP denoiser on Gaussian data using Binxu's exact recipe.

Clones `DiffusionLearningCurve/experiment/MLP_unet_learn_curve_CLI.py` and
`notebook/Figure_nn_Gaussian_learning_curve.ipynb` from Binxu's repo, with
one swap: the Karras EDM loss weight is replaced by w(σ) = σ^β so we can
sweep β and validate the spectral-bias power law.

What matches Binxu:
  - Architecture: UNetBlockStyleMLP backbone + EDM preconditioning
    (EDMMLPDenoiser is mathematically identical to Binxu's EDMPrecondWrapper).
  - σ sampling: continuous LogNormal(P_mean=-1.2, P_std=1.2).
  - σ ODE range for sampling: [0.002, 80] with ρ=7 (EDM Karras schedule).
  - Optimizer: Adam, lr=1e-4.
  - Training loop: one batch per step, torch.randint indexing, 5000 steps.
  - batch_size=1024, nhidden=256, nlayers=5, time_embed_dim=256.
  - sigma_data=0.5 (EDM default).
  - Fixed 10,000-sample Gaussian dataset.

What differs from Binxu:
  - Weight: σ^β (mean-normalized via analytic LogNormal moment), not
    Karras's (σ²+σ_d²)/(σ·σ_d)².
  - Spectrum: selectable. Default is Binxu's i.i.d. LogNormal eigenvalues
    (matches his notebook). Power-law spectrum also available for
    comparison with existing sweep.
  - No rotation: data is diagonal-Gaussian (per-coordinate variance = per-
    eigenmode variance). Rotation is not needed for spectral-bias tests.

Example (single β):
    python -m loss_weighting_spectral_bias.step1_validation.run_mlp_sweep_binxu \
        --beta 0.0 --ndim 20

Example (full sweep, one process per β):
    for B in -2 -1 0 1 2; do
        python -m loss_weighting_spectral_bias.step1_validation.run_mlp_sweep_binxu \
            --beta $B --ndim 20 --seed 42 &
    done; wait
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .losses import BetaPowerEDMLoss
from .models import EDMMLPDenoiser
from .sampling import generated_variance_per_mode
from .theory import (compute_emergence_times,
                     compute_emergence_times_relative,
                     compute_sharedW_lognormal,
                     compute_sharedW_lognormal_trajectory,
                     fit_power_law)


# Binxu's constants from:
#   DiffusionLearningCurve/notebook/Figure_nn_Gaussian_learning_curve.ipynb
#   DiffusionLearningCurve/core/diffusion_edm_lib.py
BINXU = dict(
    nlayers=5, nhidden=256, time_embed_dim=256,
    sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, rho=7.0,
    P_mean=-1.2, P_std=1.2,
    lr=1e-4, n_steps=5000, batch_size=1024,
    num_ode_steps=40, n_samples=10000,
)


def make_lognormal_eigenvalues(ndim, seed=0):
    """Binxu's Gaussian recipe: λ_k = exp(randn), i.i.d. LogNormal(0, 1)."""
    g = torch.Generator().manual_seed(seed)
    return torch.exp(torch.randn(ndim, generator=g)).numpy().astype(np.float64)


def make_power_law_eigenvalues(ndim, alpha_data):
    """Backup spectrum: λ_k = k^(-α), matches existing sweep code."""
    return np.arange(1, ndim + 1, dtype=np.float64) ** (-alpha_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--ndim", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--spectrum", choices=["lognormal", "power_law"],
                    default="lognormal",
                    help="lognormal = Binxu's i.i.d. exp(randn); "
                         "power_law = k^(-α_data) (for comparison).")
    ap.add_argument("--alpha_data", type=float, default=1.0,
                    help="Used only when --spectrum=power_law.")

    # Binxu-recipe defaults. Override only if you know why.
    ap.add_argument("--n_samples", type=int, default=BINXU["n_samples"])
    ap.add_argument("--batch_size", type=int, default=BINXU["batch_size"])
    ap.add_argument("--lr", type=float, default=BINXU["lr"])
    ap.add_argument("--n_steps", type=int, default=BINXU["n_steps"])
    ap.add_argument("--nlayers", type=int, default=BINXU["nlayers"])
    ap.add_argument("--nhidden", type=int, default=BINXU["nhidden"])
    ap.add_argument("--time_embed_dim", type=int, default=BINXU["time_embed_dim"])
    ap.add_argument("--sigma_data", type=float, default=BINXU["sigma_data"])
    ap.add_argument("--sigma_min", type=float, default=BINXU["sigma_min"])
    ap.add_argument("--sigma_max", type=float, default=BINXU["sigma_max"])
    ap.add_argument("--rho", type=float, default=BINXU["rho"])
    ap.add_argument("--P_mean", type=float, default=BINXU["P_mean"])
    ap.add_argument("--P_std", type=float, default=BINXU["P_std"])
    ap.add_argument("--num_ode_steps", type=int, default=BINXU["num_ode_steps"])
    ap.add_argument("--n_eval_samples", type=int, default=2000)
    ap.add_argument("--n_checkpoints", type=int, default=50)
    ap.add_argument("--normalize_weight", dest="normalize_weight",
                    action="store_true", default=True)
    ap.add_argument("--no_normalize_weight", dest="normalize_weight",
                    action="store_false")

    ap.add_argument("--output_dir", type=str, default="results_mlp_binxu")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data: diagonal Gaussian, eigenvalues from chosen spectrum ---
    if args.spectrum == "lognormal":
        eigvals = make_lognormal_eigenvalues(args.ndim, seed=args.seed)
    else:
        eigvals = make_power_law_eigenvalues(args.ndim, args.alpha_data)

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n_samples, args.ndim)) * np.sqrt(eigvals)[None, :]
    X_dev = torch.tensor(X, dtype=torch.float32, device=device)

    # --- Regime diagnostic before training ---
    regime = compute_sharedW_lognormal(
        eigvals, args.beta, P_mean=args.P_mean, P_std=args.P_std,
        normalize=args.normalize_weight,
    )
    lam_crit = regime["lambda_crit"]
    n_above = int(np.sum(eigvals > lam_crit))

    print(f"[binxu-recipe] β={args.beta} spectrum={args.spectrum} d={args.ndim}")
    print(f"  eigvals: min={eigvals.min():.3e}  max={eigvals.max():.3e}  "
          f"mean={eigvals.mean():.3e}")
    print(f"  σ sampling: LogNormal(P_mean={args.P_mean}, P_std={args.P_std})")
    print(f"  λ_crit (shared-W transition) = {lam_crit:.3e}")
    print(f"    modes above λ_crit: {n_above}/{args.ndim} "
          f"({'clean power-law regime' if n_above == args.ndim else 'mixed regime — saturated modes expected below λ_crit'})")
    print(f"  σ ODE range: [{args.sigma_min}, {args.sigma_max}], ρ={args.rho}")
    print(f"  model: nlayers={args.nlayers} nhidden={args.nhidden} "
          f"time_embed_dim={args.time_embed_dim} sigma_data={args.sigma_data}")
    print(f"  train: lr={args.lr} steps={args.n_steps} batch={args.batch_size}")

    # --- Model: Binxu's architecture + EDM preconditioning ---
    # EDMMLPDenoiser is mathematically identical to
    # EDMPrecondWrapper(UNetBlockStyleMLP_backbone_NoFirstNorm, ...).
    # The NoFirstNorm variant adds a Linear(d→nhidden) pre-projection when
    # d < nhidden; for d=20 this is one extra learned layer, for d=nhidden
    # (e.g. d=256) it is a no-op.
    model = EDMMLPDenoiser(
        ndim=args.ndim, sigma_data=args.sigma_data,
        nlayers=args.nlayers, nhidden=args.nhidden,
        time_embed_dim=args.time_embed_dim,
    ).to(device)

    # --- Loss: Binxu's EDMLoss with Karras weight → σ^β ---
    loss_fn = BetaPowerEDMLoss(
        beta=args.beta, P_mean=args.P_mean, P_std=args.P_std,
        normalize=args.normalize_weight,
    )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Checkpoint schedule: geometric in step, for log-time resolution ---
    ckpt_steps = np.unique(np.concatenate([
        [0],
        np.geomspace(1, args.n_steps - 1, args.n_checkpoints).astype(int),
    ]))
    ckpt_set = set(int(s) for s in ckpt_steps.tolist())

    var_list, ckpt_list, loss_trace = [], [], []
    t0 = time.time()

    # --- Training loop (Binxu): one batch per step, torch.randint indexing ---
    for step in range(args.n_steps):
        idx = torch.randint(0, args.n_samples, (args.batch_size,), device=device)
        loss = loss_fn(model, X_dev[idx])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % max(1, args.n_steps // 200) == 0:
            loss_trace.append((step, float(loss.item())))

        if step in ckpt_set:
            model.eval()
            var = generated_variance_per_mode(
                model, n_samples=args.n_eval_samples, ndim=args.ndim,
                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                num_ode_steps=args.num_ode_steps, device=device, rho=args.rho,
            )
            model.train()
            var_list.append(var)
            ckpt_list.append(step)
            print(f"  step {step:>6d}  loss={float(loss.item()):.4e}  "
                  f"var_mean={var.mean():.3e}")

    train_time = time.time() - t0
    var_traj = np.array(var_list)                # [T_ckpts, d]
    tau_arr = np.array(ckpt_list, dtype=np.float64)

    # --- Empirical emergence + power-law fit ---
    tau_rel = compute_emergence_times_relative(var_traj, tau_arr, eigvals)
    fit_rel = fit_power_law(tau_rel, eigvals)
    tau_fix = compute_emergence_times(var_traj, tau_arr, eigvals)
    fit_fix = fit_power_law(tau_fix, eigvals)

    # --- Theory prediction: shared-W under log-normal σ ---
    # Geometric τ sweep spanning the empirical range and beyond, so emergence
    # thresholds are reachable for all modes.
    tau_theory = np.geomspace(1e-3, 1e5, 500)
    var_theory = compute_sharedW_lognormal_trajectory(
        tau_theory, eigvals, args.beta,
        P_mean=args.P_mean, P_std=args.P_std,
        sigma_0=args.sigma_min, sigma_T=args.sigma_max,
        normalize=args.normalize_weight,
    )
    tau_theory_emerge = compute_emergence_times(var_theory, tau_theory, eigvals)
    fit_theory = fit_power_law(tau_theory_emerge, eigvals)

    result = dict(
        recipe="binxu_gaussian_mlp_v1",
        args=vars(args),
        eigvals=eigvals.tolist(),
        lambda_crit=float(lam_crit),
        n_modes_above_crit=n_above,
        var_traj=var_traj.tolist(),
        ckpt_steps=[int(s) for s in ckpt_list],
        loss_trace=loss_trace,
        train_time_s=train_time,
        # empirical
        alpha_trained_rel=fit_rel["alpha"],
        alpha_trained_rel_R2=fit_rel["R2"],
        alpha_trained_rel_n=fit_rel["n_used"],
        alpha_trained_fix=fit_fix["alpha"],
        alpha_trained_fix_R2=fit_fix["R2"],
        alpha_trained_fix_n=fit_fix["n_used"],
        # theory
        alpha_sharedW_LN=fit_theory["alpha"],
        alpha_sharedW_LN_R2=fit_theory["R2"],
        alpha_sharedW_LN_n=fit_theory["n_used"],
        # heuristic (per-σ asymptote, only valid for λ_k ≫ λ_crit)
        alpha_heuristic=1.0 + args.beta / 2.0,
    )

    fname = (f"{args.output_dir}/binxu_beta{args.beta:+.2f}"
             f"_d{args.ndim}_{args.spectrum}_seed{args.seed}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[result] β={args.beta}  (λ_crit={lam_crit:.3e})")
    print(f"  α_trained (rel)   = {fit_rel['alpha']:.3f}  "
          f"R²={fit_rel['R2']:.3f}  n={fit_rel['n_used']}")
    print(f"  α_trained (fixed) = {fit_fix['alpha']:.3f}  "
          f"R²={fit_fix['R2']:.3f}  n={fit_fix['n_used']}")
    print(f"  α_sharedW (LN)    = {fit_theory['alpha']:.3f}  "
          f"R²={fit_theory['R2']:.3f}  n={fit_theory['n_used']}")
    print(f"  α_heuristic       = {1.0 + args.beta / 2.0:.3f}  "
          f"(1+β/2, only valid for λ_k ≫ λ_crit)")
    print(f"  train time: {train_time:.1f}s")
    print(f"  saved: {fname}")


if __name__ == "__main__":
    main()
