"""Train shared MLP denoiser on diagonal-Gaussian data with power-law loss weighting.

Single MLP across all sigma. Emergence times come from ODE-sampled per-mode
variance (not a_k trick, which is specific to the per-sigma diagonal model).

Usage:
    python -m step1_validation.run_mlp_sweep --beta 0.0 --alpha_data 1.0 \
        --ndim 20 --max_steps 50000 --nhidden 256 --nlayers 4
"""

import argparse
import json
import os
import time
import numpy as np
import torch

from .config import (
    SIGMA_0, SIGMA_T, K_SIGMA, N_SAMPLES,
    ETA, Q_K, get_checkpoint_steps, make_eigenvalues,
)
from .models import MLPDenoiser, EDMMLPDenoiser
from .losses import SharedMLPPowerLawLoss
from .sampling import generated_variance_per_mode
from .theory import (
    compute_phi_per_sigma, compute_shared_w_trajectory,
    compute_emergence_times, compute_emergence_times_relative, fit_power_law,
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


def compute_theory(eigenvalues, beta, sigma_min, sigma_max, k_sigma,
                   w_max=None, normalize="mean"):
    """Return both per-sigma (phi-integral) and shared-W theory predictions.

    The per-sigma prediction assumes each sigma has independent parameters —
    it is the "ideal" case the main paper derives. Shared-W applies to shared
    weights (shared MLP in the convex/fixed-point limit). Under our MLP
    architecture, shared-W is the structurally correct comparator.
    """
    tau_theory = np.geomspace(1e-2, 1e6, 500)

    sigmas_np = np.logspace(np.log10(sigma_min), np.log10(sigma_max), k_sigma)
    raw = sigmas_np ** beta
    if w_max is not None:
        raw = np.clip(raw, 1.0 / w_max, w_max)
    if normalize == "mean":
        norm_const = raw.mean()
    elif normalize == "rms":
        norm_const = np.sqrt((raw ** 2).mean())
    else:
        norm_const = 1.0

    def w_fn(sigma):
        val = sigma ** beta
        if w_max is not None:
            val = np.clip(val, 1.0 / w_max, w_max)
        return val / norm_const

    # --- per-sigma (phi integral, Gauss-Legendre) ---
    var_phi = compute_phi_per_sigma(tau_theory, eigenvalues, q_k=Q_K, eta=ETA,
                                    w_fn=w_fn, sigma_0=sigma_min,
                                    sigma_T=sigma_max, n_quad=100)
    tau_phi = compute_emergence_times(var_phi, tau_theory, eigenvalues)
    fit_phi = fit_power_law(tau_phi, eigenvalues)

    # --- shared-W (A_k = E_sigma[w * (lam + sigma^2)]) ---
    w_vals = raw / norm_const  # matches loss-side normalization
    var_shared = compute_shared_w_trajectory(
        tau_theory, eigenvalues, w_vals, sigmas_np,
        sigma_0=sigma_min, sigma_T=sigma_max, eta=ETA,
    )
    tau_shared = compute_emergence_times(var_shared, tau_theory, eigenvalues)
    fit_shared = fit_power_law(tau_shared, eigenvalues)

    return {
        "alpha_heuristic": 1.0 + beta / 2.0,
        # per-sigma
        "alpha_phi": fit_phi["alpha"],
        "alpha_phi_R2": fit_phi["R2"],
        "alpha_phi_n_used": fit_phi["n_used"],
        "emergence_times_phi": np.where(np.isnan(tau_phi), None, tau_phi).tolist(),
        # shared-W (structurally correct for shared MLP)
        "alpha_sharedW": fit_shared["alpha"],
        "alpha_sharedW_R2": fit_shared["R2"],
        "alpha_sharedW_n_used": fit_shared["n_used"],
        "emergence_times_sharedW": np.where(np.isnan(tau_shared), None,
                                            tau_shared).tolist(),
    }


def build_model(args, sigma_data):
    if args.model_type == "pure":
        return MLPDenoiser(
            ndim=args.ndim, nlayers=args.nlayers, nhidden=args.nhidden,
            time_embed_dim=args.time_embed_dim,
        )
    if args.model_type == "edm":
        return EDMMLPDenoiser(
            ndim=args.ndim, sigma_data=sigma_data,
            nlayers=args.nlayers, nhidden=args.nhidden,
            time_embed_dim=args.time_embed_dim,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


def train(args, X, eigenvalues, sigma_data, device):
    model = build_model(args, sigma_data).to(device)

    loss_fn = SharedMLPPowerLawLoss(
        beta=args.beta, K_sigma=args.k_sigma,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        w_max=args.w_max, normalize=args.weight_norm,
    )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    X_dev = X.to(device)
    N = X_dev.shape[0]

    def lr_at(step):
        if args.warmup_steps <= 0 or step >= args.warmup_steps:
            return args.lr
        return args.lr * (step + 1) / args.warmup_steps

    ckpt_steps = get_checkpoint_steps(args.max_steps, args.n_checkpoints)
    ckpt_set = set(ckpt_steps.tolist())

    var_list, ckpt_list, loss_list = [], [], []
    max_grad = 0.0
    t0 = time.time()

    for step in range(args.max_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        idx = torch.randint(0, N, (args.batch_size,), device=device)
        xb = X_dev[idx]

        loss = loss_fn(model, xb)
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        max_grad = max(max_grad, float(gn))
        optimizer.step()

        if step % 50 == 0:
            loss_list.append(float(loss.item()))

        if step in ckpt_set:
            model.eval()
            var_per_mode = generated_variance_per_mode(
                model, n_samples=args.n_eval_samples, ndim=args.ndim,
                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                num_ode_steps=args.num_ode_steps, device=device,
            )
            model.train()
            var_list.append(var_per_mode)
            ckpt_list.append(step)
            print(f"  step {step:>7d}  loss={float(loss.item()):.4f}  "
                  f"var_mean={var_per_mode.mean():.3e}  gn_max={max_grad:.2f}")

    train_time = time.time() - t0
    var_traj = np.array(var_list)
    ckpt_arr = np.array(ckpt_list, dtype=np.float64)

    # Relative-progress emergence (primary metric for empirical trajectories):
    # progress = (v(t)-v(0)) / (v(inf)-v(0)), threshold 0.5. Uses empirical
    # v(inf) = last checkpoint. Robust to the random-init sampling artifact
    # where small-lambda modes have threshold already exceeded at step 0.
    tau_rel = compute_emergence_times_relative(var_traj, ckpt_arr, eigenvalues)
    fit_rel = fit_power_law(tau_rel, eigenvalues)

    # Fixed-fraction emergence (retained for parity with theory comparison).
    tau_fixed = compute_emergence_times(var_traj, ckpt_arr, eigenvalues)
    fit_fixed = fit_power_law(tau_fixed, eigenvalues)

    return {
        "var_traj": var_traj.tolist(),
        "ckpt_steps": ckpt_list,
        # primary (relative-progress) metric
        "emergence_times_trained": np.where(np.isnan(tau_rel), None,
                                            tau_rel).tolist(),
        "alpha_trained": fit_rel["alpha"],
        "alpha_trained_R2": fit_rel["R2"],
        "alpha_trained_n_used": fit_rel["n_used"],
        "n_emerged": int(np.sum(np.isfinite(tau_rel))),
        # legacy fixed-fraction metric (for comparison with theory numbers)
        "emergence_times_fixed": np.where(np.isnan(tau_fixed), None,
                                          tau_fixed).tolist(),
        "alpha_trained_fixed": fit_fixed["alpha"],
        "alpha_trained_fixed_R2": fit_fixed["R2"],
        "alpha_trained_fixed_n_used": fit_fixed["n_used"],
        # misc
        "loss_traj": loss_list,
        "max_grad_norm": max_grad,
        "train_time_s": train_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--alpha_data", type=float, default=1.0)
    p.add_argument("--ndim", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--n_samples", type=int, default=N_SAMPLES)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--grad_clip", type=float, default=10.0,
                   help="Clip pre-optimizer grad L2 norm. MLP needs ~10+; 1.0 saturates.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear LR warmup steps. Prevents initial-descent trap.")
    p.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "sgd"],
                   help="adam flattens per-mode rates via second-moment "
                        "adaptation. sgd preserves the lambda_k-scaled "
                        "gradient structure the phi-integral theory assumes.")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="SGD momentum (ignored under Adam).")

    p.add_argument("--model_type", type=str, default="pure",
                   choices=["pure", "edm"],
                   help="pure = bare MLP backbone. edm = EDM preconditioning "
                        "(c_in, c_out, c_skip, c_noise) wrapping the backbone. "
                        "Use 'edm' for Experiment Y (realistic diffusion arch).")
    p.add_argument("--sigma_data", type=float, default=None,
                   help="Data std prior for EDM preconditioning. "
                        "If None, auto-compute as sqrt(mean(eigenvalues)).")
    p.add_argument("--nlayers", type=int, default=4)
    p.add_argument("--nhidden", type=int, default=256)
    p.add_argument("--time_embed_dim", type=int, default=64)
    p.add_argument("--k_sigma", type=int, default=K_SIGMA)

    p.add_argument("--sigma_min", type=float, default=SIGMA_0,
                   help="Override global SIGMA_0. Defaults to config value.")
    p.add_argument("--sigma_max", type=float, default=SIGMA_T,
                   help="Override global SIGMA_T. Use ~2 for shared-MLP so "
                        "sigma^2 range overlaps lambda range (else E[sigma^2] "
                        "swamps lambda, flattening A_k, suppressing emergence).")

    p.add_argument("--w_max", type=float, default=None)
    p.add_argument("--weight_norm", type=str, default="mean",
                   choices=["mean", "rms", "none"])

    p.add_argument("--n_checkpoints", type=int, default=50)
    p.add_argument("--n_eval_samples", type=int, default=2000)
    p.add_argument("--num_ode_steps", type=int, default=30)

    p.add_argument("--output_dir", type=str, default="results_mlp")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[mlp-sweep] beta={args.beta} alpha_data={args.alpha_data} "
          f"d={args.ndim} seed={args.seed} device={args.device}")
    print(f"  model: nlayers={args.nlayers} nhidden={args.nhidden} "
          f"embed={args.time_embed_dim}")
    print(f"  sigma: [{args.sigma_min:g}, {args.sigma_max:g}]  "
          f"K={args.k_sigma}")
    print(f"  loss:  w_max={args.w_max} norm={args.weight_norm}")

    X, eigenvalues = generate_data(args.alpha_data, args.ndim, args.n_samples,
                                   args.seed)

    sigma_data = (args.sigma_data if args.sigma_data is not None
                  else float(np.sqrt(eigenvalues.mean())))
    print(f"  sigma_data={sigma_data:.4f}  "
          f"({'user-set' if args.sigma_data is not None else 'auto'})")

    print("Theory...")
    theory = compute_theory(eigenvalues, args.beta,
                            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            k_sigma=args.k_sigma, w_max=args.w_max,
                            normalize=args.weight_norm)
    print(f"  heuristic={_fmt(theory['alpha_heuristic'])}  "
          f"per_sigma_phi={_fmt(theory['alpha_phi'])} "
          f"(R2={_fmt(theory['alpha_phi_R2'])})  "
          f"shared_W={_fmt(theory['alpha_sharedW'])} "
          f"(R2={_fmt(theory['alpha_sharedW_R2'])})")

    print(f"Training {args.max_steps} steps "
          f"(model={args.model_type}, opt={args.optimizer}, "
          f"lr={args.lr}"
          + (f", mom={args.momentum}" if args.optimizer == "sgd" else "")
          + ")...")
    trained = train(args, X, eigenvalues, sigma_data, args.device)
    print(f"  alpha_trained (relative)   = {_fmt(trained['alpha_trained'])} "
          f"(R2={_fmt(trained['alpha_trained_R2'])}, "
          f"emerged={trained['n_emerged']}/{args.ndim})")
    print(f"  alpha_trained (fixed-frac) = "
          f"{_fmt(trained['alpha_trained_fixed'])} "
          f"(R2={_fmt(trained['alpha_trained_fixed_R2'])}, "
          f"n_used={trained['alpha_trained_fixed_n_used']}/{args.ndim})")
    print(f"  time={trained['train_time_s']:.0f}s")

    result = {
        "config": vars(args),
        "sigma_data_used": sigma_data,
        "eigenvalues": eigenvalues.tolist(),
        **theory,
        **trained,
    }

    fname = (f"mlp_{args.model_type}_{args.optimizer}"
             f"_b{args.beta:+.1f}_a{args.alpha_data:.2f}"
             f"_d{args.ndim}_h{args.nhidden}_l{args.nlayers}_s{args.seed}.json")
    path = os.path.join(args.output_dir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
