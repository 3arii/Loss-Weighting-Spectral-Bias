"""Train MLPDenoiser on synthetic DINOv2 Gaussian data.

Experiment:
  - Dataset: Gaussian X ~ N(mu, Sigma) with DINOv2 norm/patch mean+covariance
  - Model:   MLPDenoiser(d=768, nhidden, nlayers) — sigma-conditioned, no EDM preconditioning
  - Loss:    DeterministicPowerLawLoss(beta) with K=100 log-spaced sigmas per step
  - Eval:    log-scale sampling callbacks — save samples_step_{step:06d}.pt every checkpoint

Usage:
    python -m step1_validation.run_mlp_dinov2 \\
        --beta 0.0 --output_dir $STORE_DIR/step1_results/mlp_dinov2 \\
        --seed 42 --device cuda
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from torch.optim import Adam
from tqdm.auto import trange

_DLC_PATH = "/n/home12/binxuwang/Github/DiffusionLearningCurve"
if _DLC_PATH not in sys.path:
    sys.path.insert(0, _DLC_PATH)
from core.diffusion_edm_lib import edm_sampler   # noqa: E402

from .config import (
    SIGMA_0, SIGMA_T, LR, MAX_STEPS, N_SAMPLES,
    GRAD_CLIP_NORM, K_SIGMA,
)
from .models import MLPDenoiser
from .losses import DeterministicPowerLawLoss
from .dinov2_gaussian_dataset import make_dinov2_gaussian_dataset


# ---------------------------------------------------------------------------
# Log-scale callback step grid (mirrors generate_ckpt_step_list in DLC)
# ---------------------------------------------------------------------------

def generate_callback_steps(max_steps: int, num_steps: int = 200) -> list:
    """Geometrically-spaced step indices, always including step 0."""
    steps = np.geomspace(1, max_steps, num_steps).astype(int)
    steps = np.unique(steps)
    steps = steps[steps <= max_steps]
    steps = np.concatenate([[0], steps])
    return sorted(set(int(s) for s in steps))


# ---------------------------------------------------------------------------
# Sampling callback
# ---------------------------------------------------------------------------

def make_sampling_callback(sample_dir: str, ndim: int,
                            n_eval: int = 2000, num_ode_steps: int = 50,
                            device: str = "cpu"):
    """Returns a callback(step, loss, model) that generates + saves samples."""
    os.makedirs(sample_dir, exist_ok=True)
    noise_fixed = torch.randn(n_eval, ndim, generator=torch.Generator().manual_seed(0))

    def callback(step: int, loss: float, model):
        model.eval()
        with torch.no_grad():
            latents = noise_fixed.to(device)
            x_out = edm_sampler(
                model, latents,
                num_steps=num_ode_steps,
                sigma_min=model.sigma_min,
                sigma_max=model.sigma_max,
                rho=7,
                return_traj=False,
            )
        torch.save(x_out.cpu(), os.path.join(sample_dir, f"samples_step_{step:06d}.pt"))
        model.train()

    return callback


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, loss_fn, X_train, callback_steps, callback_fn,
          lr=LR, max_steps=MAX_STEPS, batch_size=512,
          grad_clip_norm=GRAD_CLIP_NORM, device="cpu"):
    """Simple training loop with Adam + grad clipping + log-scale callbacks."""
    model = model.to(device)
    X_train = X_train.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    callback_set = set(callback_steps)

    loss_traj = []
    pbar = trange(max_steps, desc="training")
    model.train()

    for step in pbar:
        idx = torch.randint(0, X_train.shape[0], (batch_size,), device=device)
        X_batch = X_train[idx]

        loss = loss_fn(model, X_batch)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        loss_val = loss.item()
        loss_traj.append(loss_val)
        pbar.set_description(f"step {step} loss {loss_val:.4f}")

        if step in callback_set:
            callback_fn(step=step, loss=loss_val, model=model)

    # Always run callback at the final step
    if (max_steps - 1) not in callback_set:
        callback_fn(step=max_steps - 1, loss=loss_traj[-1], model=model)

    model.eval()
    return loss_traj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLPDenoiser on DINOv2 Gaussian")
    parser.add_argument("--beta", type=float, required=True,
                        help="Power-law loss exponent w(sigma)=sigma^beta")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(
                            os.environ.get("STORE_DIR", "/tmp"),
                            "step1_results/mlp_dinov2"),
                        help="Root output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES,
                        help="Number of synthetic Gaussian training samples")
    parser.add_argument("--nhidden", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--n_callback_steps", type=int, default=200,
                        help="Number of log-scale callback checkpoints")
    parser.add_argument("--n_eval_samples", type=int, default=2000,
                        help="Samples to generate per callback")
    parser.add_argument("--num_ode_steps", type=int, default=50)
    parser.add_argument("--k_sigma", type=int, default=100,
                        help="Number of sigma grid points in loss")
    parser.add_argument("--layer", type=str, default="norm",
                        help="DINOv2 layer for covariance (default: norm)")
    parser.add_argument("--token", type=str, default="patch",
                        help="DINOv2 token type (default: patch)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    beta_str = f"beta{args.beta:+.1f}".replace("+", "p").replace("-", "m").replace(".", "")
    exp_name = (f"{args.layer}_{args.token}_{beta_str}_"
                f"nhid{args.nhidden}_nl{args.nlayers}_seed{args.seed}")
    exp_dir = os.path.join(args.output_dir, exp_name)
    sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {exp_dir}")
    print(f"Device:     {args.device}")

    # Save config
    cfg = vars(args)
    cfg["exp_name"] = exp_name
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Dataset
    print(f"Loading DINOv2 Gaussian dataset ({args.layer}/{args.token}, "
          f"n={args.n_samples})...")
    ds = make_dinov2_gaussian_dataset(
        n_samples=args.n_samples,
        layer=args.layer,
        token=args.token,
        seed=args.seed,
        device="cpu",
    )
    X_train = ds["X_train"]      # [N, 768]
    eigval  = ds["eigval"]       # [768] descending
    ndim    = X_train.shape[1]
    lambda_max = float(eigval[0])
    print(f"X_train: {X_train.shape}, lambda_max={lambda_max:.4f}, "
          f"lambda_min={float(eigval[-1]):.6f}")

    # Save eigenbasis for analysis
    torch.save({"eigval": eigval, "eigvec": ds["eigvec"], "mean": ds["mean"],
                "layer": args.layer, "token": args.token},
               os.path.join(exp_dir, "eigenbasis.pt"))

    # Model
    model = MLPDenoiser(
        ndim=ndim,
        nhidden=args.nhidden,
        nlayers=args.nlayers,
        time_embed_dim=args.time_embed_dim,
        sigma_min=SIGMA_0,
        sigma_max=SIGMA_T,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLPDenoiser: {n_params:,} parameters")

    # Loss
    loss_fn = DeterministicPowerLawLoss(
        beta=args.beta,
        K_sigma=args.k_sigma,
        sigma_min=SIGMA_0,
        sigma_max=SIGMA_T,
        lambda_max=lambda_max,
    )
    print(f"Loss: DeterministicPowerLawLoss(beta={args.beta}, K={args.k_sigma})")

    # Callback steps (log-scale)
    callback_steps = generate_callback_steps(args.max_steps, args.n_callback_steps)
    print(f"Callback checkpoints: {len(callback_steps)} steps "
          f"(from {callback_steps[1]} to {callback_steps[-1]})")

    callback_fn = make_sampling_callback(
        sample_dir=sample_dir,
        ndim=ndim,
        n_eval=args.n_eval_samples,
        num_ode_steps=args.num_ode_steps,
        device=args.device,
    )

    # Train
    print("Training...")
    loss_traj = train(
        model=model,
        loss_fn=loss_fn,
        X_train=X_train,
        callback_steps=callback_steps,
        callback_fn=callback_fn,
        lr=LR,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_clip_norm=GRAD_CLIP_NORM,
        device=args.device,
    )

    # Save loss trajectory and final model
    np.save(os.path.join(exp_dir, "loss_traj.npy"), np.array(loss_traj))
    torch.save(model.state_dict(), os.path.join(exp_dir, "model_final.pt"))
    np.save(os.path.join(exp_dir, "callback_steps.npy"), np.array(callback_steps))
    print(f"Done. Final loss: {loss_traj[-1]:.4f}")
    print(f"Saved {len(os.listdir(sample_dir))} sample files to {sample_dir}")


if __name__ == "__main__":
    main()
