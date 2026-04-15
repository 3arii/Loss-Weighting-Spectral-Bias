"""Simple gradient optimization sanity check for different beta schedules.

Low-dimensional (d=50) synthetic power-law Gaussian, no ODE sampling.
Instead of generating samples, directly tracks the linear denoiser
solution a_k(t) = diag(W) per eigenmode — analytically tractable.

Uses MLPDenoiser (same as main experiment) but on d=50 so training is fast.
Tracks loss per step and directly computes sample covariance every N steps
by projecting model Jacobian (for linear case) OR by small eval forward passes.

For the MLP: at each checkpoint, run a mini forward pass on a fresh
Gaussian sample batch, compute covariance, project to eigenbasis.

Usage:
    python -m step1_validation.run_simple_test --device cuda
    python -m step1_validation.run_simple_test --device cpu
"""

import sys, os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import trange
from torch.optim import Adam

sys.path.insert(0, "/n/home12/binxuwang/Github/DiffusionLearningCurve")
from core.diffusion_edm_lib import edm_sampler  # noqa

from .config import SIGMA_0, SIGMA_T, GRAD_CLIP_NORM
from .models import MLPDenoiser
from .losses import DeterministicPowerLawLoss

# ── Synthetic dataset ────────────────────────────────────────────────────────

def make_synthetic_gaussian(d=50, alpha_data=1.0, n_samples=5000, seed=42):
    """Power-law eigenspectrum: lambda_k = k^{-alpha_data}."""
    rng = torch.Generator().manual_seed(seed)
    k = torch.arange(1, d + 1, dtype=torch.float32)
    eigval = k ** (-alpha_data)                    # [d] descending(ish)
    # Build random orthogonal eigvec via QR
    A = torch.randn(d, d, generator=rng)
    eigvec, _ = torch.linalg.qr(A)                # [d, d]
    # Sample X ~ N(0, U diag(eigval) U^T)
    z = torch.randn(n_samples, d, generator=rng)   # [n, d]
    X = z * eigval.sqrt().unsqueeze(0)             # in eigenbasis
    X = X @ eigvec.T                               # rotate to data space
    return X, eigval.numpy(), eigvec

# ── Training with covariance tracking ────────────────────────────────────────

@torch.no_grad()
def eval_covariance(model, eigvec, d, n_eval=2000, device="cpu"):
    """Generate samples via edm_sampler, compute var_k in eigenbasis."""
    model.eval()
    latents = torch.randn(n_eval, d, device=device)
    x_gen = edm_sampler(model, latents, num_steps=20,
                        sigma_min=model.sigma_min, sigma_max=model.sigma_max,
                        rho=7, return_traj=False)
    cov = torch.cov(x_gen.cpu().T)
    var_k = (eigvec.T @ cov @ eigvec).diag().numpy()
    model.train()
    return var_k

def train_and_track(beta, eigval, eigvec, X_train,
                    d=50, max_steps=5000, batch_size=256,
                    eval_every=100, device="cpu"):
    """Train MLPDenoiser(d) and track loss + covariance every eval_every steps."""
    lambda_max = float(eigval[0])
    model = MLPDenoiser(ndim=d, nhidden=128, nlayers=3,
                        time_embed_dim=32,
                        sigma_min=SIGMA_0, sigma_max=SIGMA_T).to(device)
    loss_fn = DeterministicPowerLawLoss(beta=beta, K_sigma=50,
                                        sigma_min=SIGMA_0, sigma_max=SIGMA_T,
                                        lambda_max=lambda_max)
    optimizer = Adam(model.parameters(), lr=0.01)
    X_train = X_train.to(device)

    steps_log, loss_log, var_traj = [], [], []

    eval_steps = set(range(0, max_steps, eval_every)) | {max_steps - 1}
    model.train()
    pbar = trange(max_steps, desc=f"β={beta:+.1f}", leave=False)

    for step in pbar:
        idx = torch.randint(0, X_train.shape[0], (batch_size,), device=device)
        loss = loss_fn(model, X_train[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        loss_val = loss.item()
        pbar.set_description(f"β={beta:+.1f} loss={loss_val:.4f}")

        if step in eval_steps:
            steps_log.append(step)
            loss_log.append(loss_val)
            var_k = eval_covariance(model, eigvec, d, n_eval=1000, device=device)
            var_traj.append(var_k)

    return {
        "steps":    np.array(steps_log),
        "loss":     np.array(loss_log),
        "var_traj": np.stack(var_traj, axis=0),   # [n_evals, d]
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d",          type=int,   default=50)
    p.add_argument("--alpha_data", type=float, default=1.0)
    p.add_argument("--max_steps",  type=int,   default=5000)
    p.add_argument("--eval_every", type=int,   default=100)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="/tmp/simple_test")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {args.device}  d={args.d}  alpha_data={args.alpha_data}")

    X_train, eigval, eigvec = make_synthetic_gaussian(
        d=args.d, alpha_data=args.alpha_data, n_samples=5000)

    betas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    colors = {-2.0: "#d62728", -1.0: "#ff7f0e", 0.0: "#2ca02c",
               1.0: "#1f77b4",  2.0: "#9467bd"}
    results = {}

    for beta in betas:
        print(f"\nTraining β={beta:+.1f}...")
        res = train_and_track(beta, eigval, eigvec, X_train,
                              d=args.d, max_steps=args.max_steps,
                              eval_every=args.eval_every, device=args.device)
        results[beta] = res
        print(f"  Final loss: {res['loss'][-1]:.4f}")

    # ── Plot 1: Loss curves ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for beta, res in results.items():
        ax.semilogy(res["steps"], res["loss"],
                    color=colors[beta], label=f"β={beta:+.1f}", lw=1.8)
    ax.set_xlabel("Training step"); ax.set_ylabel("Loss (log)")
    ax.set_title(f"Loss — d={args.d}, α_data={args.alpha_data}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/loss_curves.png", dpi=150)
    plt.close()

    # ── Plot 2: Covariance heatmap ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (beta, res) in zip(axes, results.items()):
        ratio = np.clip(res["var_traj"].T / (eigval[:, None] + 1e-10), 0, 1.5)
        im = ax.imshow(ratio, aspect="auto", origin="upper",
                       vmin=0, vmax=1.2, cmap="RdYlGn",
                       extent=[res["steps"][0], res["steps"][-1], args.d, 0])
        ax.axhline(0, color='white', lw=0.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Mode k")
        ax.set_title(f"β={beta:+.1f}")
        plt.colorbar(im, ax=ax, label="var_k/λ_k")
    axes[-1].axis("off")
    plt.suptitle(f"Covariance convergence — d={args.d}, α_data={args.alpha_data}", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/cov_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Convergence lines ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    mode_idx = np.unique(np.geomspace(1, args.d - 1, 8).astype(int))
    cmap = plt.cm.plasma
    for ax, (beta, res) in zip(axes, results.items()):
        for i, k in enumerate(mode_idx):
            ratio_k = res["var_traj"][:, k] / (eigval[k] + 1e-10)
            ax.plot(res["steps"], ratio_k, color=cmap(i / len(mode_idx)),
                    lw=1.5, label=f"k={k} λ={eigval[k]:.3f}")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.axhline(0.5, color="red",  ls=":",  lw=1)
        ax.set_xlabel("Step"); ax.set_ylabel("var_k/λ_k")
        ax.set_title(f"β={beta:+.1f}"); ax.set_ylim(-0.05, 1.6)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]: ax.legend(fontsize=7)
    axes[-1].axis("off")
    plt.suptitle(f"Eigenmode convergence — d={args.d}, α_data={args.alpha_data}")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/cov_lines.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save arrays to disk ──────────────────────────────────────────────────
    for beta, res in results.items():
        tag = f"beta{beta:+.1f}".replace("+", "p").replace("-", "m").replace(".", "")
        np.save(f"{args.output_dir}/steps_{tag}.npy",    res["steps"])
        np.save(f"{args.output_dir}/loss_{tag}.npy",     res["loss"])
        np.save(f"{args.output_dir}/var_traj_{tag}.npy", res["var_traj"])

    # ── Plot 4: Zoomed early-steps convergence (left=full, right=first 500) ─
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    for row, (beta, res) in enumerate(results.items()):
        steps, var_traj = res["steps"], res["var_traj"]
        for col, (xlim, suffix) in enumerate([(None, "full 5000 steps"),
                                               (500,  "zoomed: first 500 steps")]):
            ax = axes[row, col]
            for i, k in enumerate(mode_idx):
                ratio_k = var_traj[:, k] / (eigval[k] + 1e-10)
                ax.plot(steps, ratio_k, color=cmap(i / len(mode_idx)),
                        lw=1.5, label=f"k={k} λ={eigval[k]:.3f}", alpha=0.85)
            ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
            ax.axhline(0.5, color="red",  ls=":",  lw=1, alpha=0.6)
            ax.set_xlabel("Step"); ax.set_ylabel("var_k / λ_k")
            ax.set_title(f"β={beta:+.1f} — {suffix}")
            ax.set_ylim(-0.05, 1.8); ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(0, xlim)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2)
    plt.suptitle(f"Eigenmode convergence — d={args.d}, α_data={args.alpha_data}\n"
                 "Left: full run | Right: zoomed early", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/cov_lines_zoomed.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 5: Log-scale step axis ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (beta, res) in zip(axes, results.items()):
        steps, var_traj = res["steps"], res["var_traj"]
        for i, k in enumerate(mode_idx):
            ratio_k = var_traj[:, k] / (eigval[k] + 1e-10)
            ax.plot(steps + 1, ratio_k, color=cmap(i / len(mode_idx)),
                    lw=1.5, alpha=0.85, label=f"k={k}")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.axhline(0.5, color="red",  ls=":",  lw=1)
        ax.set_xscale("log"); ax.set_xlabel("Step (log)"); ax.set_ylabel("var_k/λ_k")
        ax.set_title(f"β={beta:+.1f}"); ax.set_ylim(-0.05, 1.8); ax.grid(True, alpha=0.3)
        if ax is axes[0]: ax.legend(fontsize=7, ncol=2)
    axes[-1].axis("off")
    plt.suptitle(f"Eigenmode convergence (log-step) — d={args.d}, α_data={args.alpha_data}")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/cov_lines_logstep.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll plots saved to {args.output_dir}")

    # Summary table
    print("\n  β    | final_loss | modes_converged(>0.5)")
    print("  -----|------------|----------------------")
    for beta, res in results.items():
        final_ratio = res["var_traj"][-1] / (eigval + 1e-10)
        n_conv = int((final_ratio > 0.5).sum())
        print(f"  {beta:+.1f}  | {res['loss'][-1]:10.4f} | {n_conv}/{args.d}")


if __name__ == "__main__":
    main()
