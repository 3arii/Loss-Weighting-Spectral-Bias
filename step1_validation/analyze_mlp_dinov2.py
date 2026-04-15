"""Posthoc spectral analysis for MLPDenoiser DINOv2 Gaussian experiments.

Loads saved samples_step_*.pt files from run_mlp_dinov2.py, computes the
covariance trajectory in the data eigenbasis, finds emergence steps, and
fits the power-law exponent alpha.

Usage:
    python -m step1_validation.analyze_mlp_dinov2 \\
        --exp_dir $STORE_DIR/step1_results/mlp_dinov2/norm_patch_beta0p0_nhid512_nl6_seed42 \\
        --output_dir $STORE_DIR/step1_results/mlp_dinov2_analysis

    # Batch mode: analyze all experiments in a root directory
    python -m step1_validation.analyze_mlp_dinov2 \\
        --root_dir $STORE_DIR/step1_results/mlp_dinov2 \\
        --output_dir $STORE_DIR/step1_results/mlp_dinov2_analysis
"""

import sys
import os
import glob
import json
import argparse
import numpy as np
import torch
import pandas as pd

_DLC_PATH = "/n/home12/binxuwang/Github/DiffusionLearningCurve"
if _DLC_PATH not in sys.path:
    sys.path.insert(0, _DLC_PATH)
from core.trajectory_convergence_lib import (   # noqa: E402
    compute_crossing_points,
    fit_regression_log_scale,
)

from .config import THRESHOLD_FRAC


# ---------------------------------------------------------------------------
# Load samples from experiment directory
# ---------------------------------------------------------------------------

def load_sample_files(sample_dir: str):
    """Load all samples_step_*.pt files, sorted by step.

    Returns:
        steps:    list of int step indices
        samples:  list of torch.Tensor [N, d]
    """
    pattern = os.path.join(sample_dir, "samples_step_*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No sample files found in {sample_dir}")

    steps, samples = [], []
    for p in paths:
        fname = os.path.basename(p)
        step = int(fname.replace("samples_step_", "").replace(".pt", ""))
        x = torch.load(p, weights_only=True)
        steps.append(step)
        samples.append(x.float())

    return steps, samples


# ---------------------------------------------------------------------------
# Covariance trajectory in eigenbasis
# ---------------------------------------------------------------------------

def compute_var_trajectory(samples: list, eigvec: torch.Tensor):
    """Project each sample batch covariance into eigenbasis.

    Args:
        samples:  list of [N, d] tensors, one per checkpoint
        eigvec:   [d, d] eigenvectors (columns), data eigenbasis

    Returns:
        var_traj: np.ndarray [n_ckpts, d]
    """
    var_list = []
    for x in samples:
        x = x.float()
        cov = torch.cov(x.T)                    # [d, d]
        cov_proj = eigvec.T @ cov @ eigvec      # [d, d]
        var_k = cov_proj.diag().numpy()         # [d]
        var_list.append(var_k)
    return np.stack(var_list, axis=0)           # [n_ckpts, d]


# ---------------------------------------------------------------------------
# Single experiment analysis
# ---------------------------------------------------------------------------

def analyze_experiment(exp_dir: str, output_dir: str = None,
                       threshold_frac: float = THRESHOLD_FRAC,
                       threshold_type: str = "harmonic_mean") -> dict:
    """Full analysis pipeline for one experiment directory.

    Returns dict with: beta, alpha, R2, n_emerged, n_modes, df
    """
    # Load config
    cfg_path = os.path.join(exp_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    beta = cfg["beta"]
    exp_name = cfg.get("exp_name", os.path.basename(exp_dir))

    # Load eigenbasis
    eb = torch.load(os.path.join(exp_dir, "eigenbasis.pt"), weights_only=True)
    eigval = eb["eigval"].float()   # [d] descending
    eigvec = eb["eigvec"].float()   # [d, d]

    # Load samples
    sample_dir = os.path.join(exp_dir, "samples")
    steps, samples = load_sample_files(sample_dir)
    print(f"  Loaded {len(steps)} checkpoints, "
          f"steps {steps[0]}..{steps[-1]}, "
          f"samples shape {samples[0].shape}")

    # Compute covariance trajectory
    var_traj = compute_var_trajectory(samples, eigvec)    # [n_ckpts, d]

    # Emergence detection
    step_arr = np.array(steps, dtype=float)
    df = compute_crossing_points(
        target_eigval      = eigval,
        empiric_var_traj   = var_traj,
        step_slice         = step_arr,
        threshold_type     = threshold_type,
        smooth_sigma       = 1.0,
        threshold_fraction = threshold_frac,
    )

    df_valid = df[df["emergence_step"].notna()].copy()
    n_emerged = len(df_valid)
    n_modes = len(eigval)
    print(f"  beta={beta:+.1f}: {n_emerged}/{n_modes} modes emerged")

    result = {
        "exp_name":  exp_name,
        "beta":      beta,
        "alpha":     float("nan"),
        "R2":        float("nan"),
        "n_emerged": n_emerged,
        "n_modes":   n_modes,
        "df":        df,
    }

    if n_emerged >= 2:
        fit = fit_regression_log_scale(
            df_valid["Variance"].values,
            df_valid["emergence_step"].values,
        )
        result["alpha"] = -fit["slope"]
        result["R2"]    = fit["r_squared"]
        result["fit"]   = fit
        print(f"  alpha={result['alpha']:.3f}, R2={result['R2']:.3f}")

    # Save outputs
    if output_dir is None:
        output_dir = os.path.join(exp_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Save covariance trajectory
    np.save(os.path.join(output_dir, f"{exp_name}_var_traj.npy"), var_traj)
    np.save(os.path.join(output_dir, f"{exp_name}_eigval.npy"), eigval.numpy())
    np.save(os.path.join(output_dir, f"{exp_name}_steps.npy"), np.array(steps))

    # Save emergence dataframe
    df.to_csv(os.path.join(output_dir, f"{exp_name}_emergence.csv"), index=False)

    # Save summary (cast numpy scalars to native Python types for JSON)
    summary = {k: (float(v) if hasattr(v, "item") else v)
               for k, v in result.items() if k not in ("df", "fit")}
    with open(os.path.join(output_dir, f"{exp_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_all(root_dir: str, output_dir: str,
                threshold_frac: float = THRESHOLD_FRAC) -> pd.DataFrame:
    """Analyze all experiment subdirectories under root_dir."""
    exp_dirs = sorted([
        d for d in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))
    ])
    print(f"Found {len(exp_dirs)} experiments in {root_dir}")

    rows = []
    for exp_dir in exp_dirs:
        print(f"\nAnalyzing: {os.path.basename(exp_dir)}")
        try:
            res = analyze_experiment(exp_dir, output_dir=output_dir,
                                     threshold_frac=threshold_frac)
            rows.append({k: v for k, v in res.items() if k not in ("df", "fit")})
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"exp_name": os.path.basename(exp_dir), "error": str(e)})

    df_all = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "summary_all.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\nSummary table saved to {out_path}")
    print(df_all.to_string(index=False))
    return df_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MLPDenoiser DINOv2 experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp_dir", type=str, help="Single experiment directory")
    group.add_argument("--root_dir", type=str, help="Root dir with multiple experiments")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write analysis outputs")
    parser.add_argument("--threshold_frac", type=float, default=THRESHOLD_FRAC)
    parser.add_argument("--threshold_type", type=str, default="harmonic_mean",
                        choices=["harmonic_mean", "mean", "geometric_mean", "range"])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.exp_dir:
        out = args.output_dir or os.path.join(args.exp_dir, "analysis")
        res = analyze_experiment(
            args.exp_dir, output_dir=out,
            threshold_frac=args.threshold_frac,
            threshold_type=args.threshold_type,
        )
        print(f"\nbeta={res['beta']:+.1f}  alpha={res['alpha']:.3f}  "
              f"R2={res['R2']:.3f}  emerged={res['n_emerged']}/{res['n_modes']}")
    else:
        out = args.output_dir or os.path.join(args.root_dir, "analysis")
        analyze_all(args.root_dir, output_dir=out,
                    threshold_frac=args.threshold_frac)


if __name__ == "__main__":
    main()
