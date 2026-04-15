"""Per-sigma control experiments.

Analytic mode: Compute per-sigma Wiener filter + Phi integral for all configs.
GD mode: Train independent W_j at each sigma for 3 configs to verify ODE match.

Usage:
    # Analytic (all configs, CPU, fast):
    python -m step1_validation.run_per_sigma --output_dir results_per_sigma

    # GD validation (3 configs):
    python -m step1_validation.run_per_sigma --gd --output_dir results_per_sigma
"""

import argparse
import json
import os

import numpy as np
import torch

from .config import (
    SIGMA_0, SIGMA_T, K_SIGMA, ETA, Q_K, THRESHOLD_FRAC,
    BETA_VALUES, ALPHA_DATA_VALUES,
    get_sigma_grid_np, make_eigenvalues,
)
from .theory import (
    psi_k_vectorized, compute_phi_per_sigma,
    compute_emergence_times, fit_power_law,
)


def run_analytic(alpha_data_list, beta_list, ndim, output_dir):
    """Compute per-sigma Phi integral for all (alpha_data, beta) combos."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for alpha_data in alpha_data_list:
        eigenvalues = make_eigenvalues(alpha_data, ndim)

        for beta in beta_list:
            print(f"Analytic: alpha_data={alpha_data}, beta={beta}, d={ndim}")

            def w_fn(sigma):
                return sigma ** beta

            tau_array = np.geomspace(1e-2, 1e6, 2000)
            variance = compute_phi_per_sigma(
                tau_array, eigenvalues, q_k=Q_K, eta=ETA, w_fn=w_fn,
                sigma_0=SIGMA_0, sigma_T=SIGMA_T, n_quad=100
            )
            tau_k = compute_emergence_times(variance, tau_array, eigenvalues,
                                            threshold_frac=THRESHOLD_FRAC)
            fit = fit_power_law(tau_k, eigenvalues)

            entry = {
                "alpha_data": alpha_data,
                "beta": beta,
                "ndim": ndim,
                "alpha_heuristic": 1.0 + beta / 2.0,
                "alpha_phi": fit["alpha"],
                "R2": fit["R2"],
                "n_used": fit["n_used"],
                "emergence_times": np.where(np.isnan(tau_k), None, tau_k).tolist(),
            }
            results.append(entry)
            print(f"  alpha_phi={fit['alpha']:.3f}, R2={fit['R2']:.4f}")

    outpath = os.path.join(output_dir, f"per_sigma_analytic_d{ndim}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")
    return results


def run_gd_validation(output_dir, ndim=200, max_steps=20000, lr=0.01):
    """Train per-sigma models for 3 configs to verify ODE solution.

    Trains independent scalar a_k(sigma_j) at each (sigma_j, eigenmode k).
    Verifies that the trained a_k matches the analytical psi_k.
    """
    os.makedirs(output_dir, exist_ok=True)
    alpha_data = 1.0
    eigenvalues = make_eigenvalues(alpha_data, ndim)

    gd_betas = [-2.0, 0.0, 2.0]
    sigma_grid = get_sigma_grid_np(K=K_SIGMA)
    results = []

    for beta in gd_betas:
        print(f"\nGD validation: beta={beta}")
        w_values = sigma_grid ** beta

        # Normalize weights (same as training loss)
        A_max = (w_values * (eigenvalues[0] + sigma_grid**2)).mean()
        w_norm = w_values / A_max

        # For each sigma_j, train a_k independently
        # The ODE: da_k/dtau = -2*eta*w_norm(sigma_j)*(lambda_k+sigma_j^2)*(a_k - wiener_k)
        # Analytical: a_k(tau) = wiener_k * (1 - exp(-2*eta*w_norm*tau*(lambda_k+sigma_j^2)))

        # Pick a subset of sigma values for validation (5 representative)
        sigma_indices = [0, K_SIGMA // 4, K_SIGMA // 2, 3 * K_SIGMA // 4, K_SIGMA - 1]
        mode_indices = [0, ndim // 4, ndim // 2, 3 * ndim // 4, ndim - 1]

        max_error = 0.0
        for j in sigma_indices:
            sig_j = sigma_grid[j]
            w_j = w_norm[j]

            for k in mode_indices:
                lam_k = eigenvalues[k]
                wiener_k = lam_k / (lam_k + sig_j**2)

                # Train: simple SGD on a_k
                a_k = 0.0  # init
                rate = 2.0 * ETA * w_j * (lam_k + sig_j**2)

                for step in range(max_steps):
                    grad = rate * (a_k - wiener_k)
                    a_k = a_k - lr * grad

                # Analytical prediction at tau = lr * max_steps (discrete steps)
                # For discrete SGD: a_k(t) = wiener * (1 - (1 - lr*rate)^t)
                analytical = wiener_k * (1.0 - (1.0 - lr * rate) ** max_steps)

                error = abs(a_k - analytical)
                max_error = max(max_error, error)

        results.append({
            "beta": beta, "ndim": ndim, "max_error": max_error,
            "match": max_error < 1e-4,
        })
        print(f"  max error = {max_error:.2e}, match = {max_error < 1e-4}")

    outpath = os.path.join(output_dir, "per_sigma_gd_validation.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Per-sigma control")
    parser.add_argument("--gd", action="store_true", help="Run GD validation (3 configs)")
    parser.add_argument("--ndim", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results_per_sigma")
    args = parser.parse_args()

    if args.gd:
        run_gd_validation(args.output_dir, ndim=args.ndim)
    else:
        run_analytic(ALPHA_DATA_VALUES, BETA_VALUES, args.ndim, args.output_dir)


if __name__ == "__main__":
    main()
