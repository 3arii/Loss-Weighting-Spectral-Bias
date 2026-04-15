"""Analyze results and generate figures for Step 1.

Loads JSON result files, aggregates across seeds, produces:
- Figure 1: Five-way alpha vs beta comparison (main result)
- Figure 2: Emergence heatmaps (beta=0 vs beta=-1)
- Figure 3: Named schemes + practitioner lookup
- Figures A1-A5: Appendix

Usage:
    python -m step1_validation.analyze_results --results_dir results --output_dir figures
"""

import argparse
import json
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .config import (
    BETA_VALUES, ALPHA_DATA_VALUES, SIGMA_0, SIGMA_T, THRESHOLD_FRAC,
    make_eigenvalues,
)


def load_results(results_dir):
    """Load all result JSONs into a list."""
    files = glob.glob(os.path.join(results_dir, "beta_*.json"))
    results = []
    for f in sorted(files):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def load_per_sigma_results(results_dir):
    """Load per-sigma analytic results."""
    files = glob.glob(os.path.join(results_dir, "per_sigma_analytic_*.json"))
    all_results = []
    for f in files:
        with open(f) as fp:
            all_results.extend(json.load(fp))
    return all_results


def aggregate_by_config(results):
    """Group results by (beta, alpha_data, ndim) and aggregate across seeds."""
    groups = {}
    for r in results:
        cfg = r["config"]
        key = (cfg["beta"], cfg["alpha_data"], cfg["ndim"])
        groups.setdefault(key, []).append(r)

    aggregated = {}
    for key, runs in groups.items():
        alphas_trained = [r["alpha_trained_shared_W"] for r in runs
                          if r["alpha_trained_shared_W"] is not None]
        alphas_phi = [r["alpha_phi_per_sigma"] for r in runs
                      if r["alpha_phi_per_sigma"] is not None]

        aggregated[key] = {
            "beta": key[0],
            "alpha_data": key[1],
            "ndim": key[2],
            "n_seeds": len(runs),
            "alpha_heuristic": runs[0]["alpha_heuristic"],
            "alpha_phi_per_sigma": runs[0]["alpha_phi_per_sigma"],
            "alpha_shared_W_theory": runs[0]["alpha_shared_W_theory"],
            "alpha_trained_mean": np.mean(alphas_trained) if alphas_trained else np.nan,
            "alpha_trained_std": np.std(alphas_trained) if alphas_trained else np.nan,
            "alpha_trained_all": alphas_trained,
            "n_inaccessible": runs[0]["n_modes_inaccessible"],
        }
    return aggregated


def figure_1_five_way_comparison(aggregated, output_dir):
    """Figure 1: alpha vs beta with all 5 comparison curves."""
    fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(10, 8),
                                             height_ratios=[3, 1],
                                             sharex=True)

    for alpha_data in ALPHA_DATA_VALUES:
        betas = sorted([k[0] for k in aggregated if k[1] == alpha_data])
        if not betas:
            continue

        a_heur = [1.0 + b / 2.0 for b in betas]
        a_phi = [aggregated[(b, alpha_data, aggregated[(b, alpha_data,
                  list(aggregated.keys())[0][2])]["ndim"])]["alpha_phi_per_sigma"]
                 if (b, alpha_data, list(aggregated.keys())[0][2]) in aggregated else np.nan
                 for b in betas]
        a_sw_th = []
        a_trained_mean = []
        a_trained_std = []

        ndim = None
        for b in betas:
            for key in aggregated:
                if key[0] == b and key[1] == alpha_data:
                    ndim = key[2]
                    break
            if ndim is None:
                continue
            entry = aggregated.get((b, alpha_data, ndim))
            if entry:
                a_sw_th.append(entry["alpha_shared_W_theory"])
                a_trained_mean.append(entry["alpha_trained_mean"])
                a_trained_std.append(entry["alpha_trained_std"])
            else:
                a_sw_th.append(np.nan)
                a_trained_mean.append(np.nan)
                a_trained_std.append(np.nan)

        label_suffix = f" (alpha_data={alpha_data})"

        # Main panel
        if alpha_data == ALPHA_DATA_VALUES[0]:  # plot heuristic once
            ax_main.plot(betas, a_heur, "k--", label="Heuristic: 1+beta/2", zorder=5)

        ax_main.plot(betas, a_phi, "o-", markersize=5,
                     label=f"Phi integral{label_suffix}", alpha=0.8)
        ax_main.plot(betas, a_sw_th, "s--", markersize=5,
                     label=f"Shared-W theory{label_suffix}", alpha=0.8)
        ax_main.errorbar(betas, a_trained_mean, yerr=a_trained_std,
                         fmt="D", markersize=6, capsize=3,
                         label=f"Shared-W trained{label_suffix}", alpha=0.9)

        # Residual panel
        resid = [tm - st if (np.isfinite(tm) and np.isfinite(st)) else np.nan
                 for tm, st in zip(a_trained_mean, a_sw_th)]
        ax_resid.plot(betas, resid, "D-", markersize=4,
                      label=f"Trained - Theory{label_suffix}", alpha=0.8)

    ax_main.set_ylabel("Emergence exponent alpha")
    ax_main.set_title("Five-Way Alpha Comparison")
    ax_main.legend(fontsize=8, ncol=2)
    ax_main.grid(True, alpha=0.3)

    ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_resid.set_xlabel("beta")
    ax_resid.set_ylabel("Residual (trained - theory)")
    ax_resid.legend(fontsize=8)
    ax_resid.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_five_way_comparison.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "fig1_five_way_comparison.pdf"))
    plt.close()
    print("Saved Figure 1")


def figure_2_emergence_heatmaps(results, output_dir, alpha_data=1.0):
    """Figure 2: Emergence heatmaps for beta=0 vs beta=-1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target_beta in zip(axes, [0.0, -1.0]):
        # Find a run with this beta and alpha_data
        run = None
        for r in results:
            if (r["config"]["beta"] == target_beta and
                r["config"]["alpha_data"] == alpha_data):
                run = r
                break

        if run is None:
            ax.text(0.5, 0.5, f"No data for beta={target_beta}",
                    transform=ax.transAxes, ha="center")
            continue

        variance_traj = np.array(run["variance_trajectories"])
        eigenvalues = np.array(run["eigenvalues"])
        steps = np.array(run["checkpoint_steps"])
        ndim = run["config"]["ndim"]

        # Normalize: lambda_tilde_k / lambda_k
        ratio = variance_traj / eigenvalues[None, :]

        im = ax.pcolormesh(
            steps, np.arange(ndim), ratio.T,
            norm=LogNorm(vmin=1e-6, vmax=10), cmap="viridis", shading="auto"
        )
        ax.contour(steps, np.arange(ndim), ratio.T,
                   levels=[THRESHOLD_FRAC], colors="white", linewidths=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Eigenmode k")
        ax.set_title(f"beta = {target_beta}")
        fig.colorbar(im, ax=ax, label="lambda_tilde_k / lambda_k")

    plt.suptitle(f"Emergence Heatmaps (alpha_data={alpha_data})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_emergence_heatmaps.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "fig2_emergence_heatmaps.pdf"))
    plt.close()
    print("Saved Figure 2")


def figure_3_practitioner_lookup(aggregated, output_dir):
    """Figure 3: Practitioner lookup — emergence time ratio vs alpha_data."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # For each beta, plot the emergence time ratio (last mode / first mode)
    # using the Phi integral
    alpha_data_range = np.linspace(0.3, 3.0, 50)
    d = 768

    for beta in [-2.0, -1.0, -0.5, 0.0, 1.0]:
        alpha_heur = 1.0 + beta / 2.0
        # Ratio = (lambda_d / lambda_1)^alpha = d^{-alpha_data * alpha}
        ratios = d ** (alpha_data_range * alpha_heur)
        ax.plot(alpha_data_range, ratios, label=f"beta={beta} (alpha={alpha_heur:.1f})")

    # Mark DINOv2 anchor
    ax.axvline(0.56, color="red", linestyle=":", alpha=0.7, label="DINOv2 (alpha_data=0.56)")

    ax.set_xlabel("Eigenvalue decay exponent (alpha_data)")
    ax.set_ylabel("Emergence time ratio (last / first mode)")
    ax.set_yscale("log")
    ax.set_title(f"Practitioner Lookup (d={d}, heuristic)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_practitioner_lookup.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "fig3_practitioner_lookup.pdf"))
    plt.close()
    print("Saved Figure 3")


def figure_a5_inaccessible_modes(aggregated, output_dir):
    """Figure A5: Fraction of inaccessible modes vs (beta, alpha_data)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for alpha_data in ALPHA_DATA_VALUES:
        betas = []
        fracs = []
        for key, entry in sorted(aggregated.items()):
            if key[1] == alpha_data:
                betas.append(key[0])
                fracs.append(entry["n_inaccessible"] / key[2])

        ax.plot(betas, fracs, "o-", label=f"alpha_data={alpha_data}")

    ax.set_xlabel("beta")
    ax.set_ylabel("Fraction inaccessible modes")
    ax.set_title("Inaccessible Modes (Shared-W Model)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figA5_inaccessible_modes.png"), dpi=150)
    plt.close()
    print("Saved Figure A5")


def print_summary_table(aggregated):
    """Print five-way comparison table to console."""
    print("\n" + "=" * 100)
    print(f"{'beta':>6} {'alpha_data':>10} {'d':>5} | "
          f"{'heuristic':>9} {'Phi':>8} {'SW_theory':>9} {'SW_trained':>10} {'+/-':>5} | "
          f"{'R2':>5} {'inacc':>5}")
    print("-" * 100)

    for key in sorted(aggregated.keys()):
        e = aggregated[key]
        print(f"{e['beta']:>6.1f} {e['alpha_data']:>10.2f} {e['ndim']:>5d} | "
              f"{e['alpha_heuristic']:>9.3f} {e['alpha_phi_per_sigma']:>8.3f} "
              f"{e['alpha_shared_W_theory']:>9.3f} {e['alpha_trained_mean']:>10.3f} "
              f"{e['alpha_trained_std']:>5.3f} | "
              f"{0.0:>5.3f} {e['n_inaccessible']:>5d}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Analyze results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--per_sigma_dir", type=str, default="results_per_sigma")
    parser.add_argument("--output_dir", type=str, default="figures/step1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result files")

    # Aggregate
    aggregated = aggregate_by_config(results)
    print_summary_table(aggregated)

    # Figures
    figure_1_five_way_comparison(aggregated, args.output_dir)
    figure_2_emergence_heatmaps(results, args.output_dir)
    figure_3_practitioner_lookup(aggregated, args.output_dir)
    figure_a5_inaccessible_modes(aggregated, args.output_dir)


if __name__ == "__main__":
    main()
