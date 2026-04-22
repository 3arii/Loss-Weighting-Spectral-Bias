"""Phi integral theory for power-law validation.

Computes exact per-sigma emergence times via Gauss-Legendre quadrature.
Also computes shared-W analytical predictions and a_k-based emergence.
All vectorized — no Python loops over tau.
"""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from sklearn.linear_model import LinearRegression

from .config import SIGMA_0, SIGMA_T, ETA, Q_K, THRESHOLD_FRAC


# --- Per-sigma ODE solution ---

def psi_k_vectorized(sigma_grid, tau_array, lambda_k_arr, q_k, eta, w_values):
    """Per-sigma ODE solution. Returns [T, K, d]."""
    T = len(tau_array)
    K = len(sigma_grid)
    d = len(lambda_k_arr)

    sig2 = sigma_grid.reshape(K, 1) ** 2    # [K, 1]
    lam = lambda_k_arr.reshape(1, d)         # [1, d]
    wiener = lam / (lam + sig2)              # [K, d]
    rate = w_values.reshape(K, 1) * (lam + sig2)  # [K, d]

    exponent = -2.0 * eta * tau_array.reshape(T, 1, 1) * rate.reshape(1, K, d)
    return wiener.reshape(1, K, d) + (q_k - wiener.reshape(1, K, d)) * np.exp(exponent)


# --- Phi integral via Gauss-Legendre ---

def compute_phi_per_sigma(tau_array, lambda_k_arr, q_k=Q_K, eta=ETA,
                          w_fn=None, sigma_0=SIGMA_0, sigma_T=SIGMA_T, n_quad=100):
    """Compute generated variance trajectories [T, d] via Gauss-Legendre quadrature."""
    nodes, weights = leggauss(n_quad)
    a, b = np.log(sigma_0), np.log(sigma_T)
    log_sigma = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    quad_weights = 0.5 * (b - a) * weights
    quad_sigma = np.exp(log_sigma)

    w_at_nodes = np.array([w_fn(s) for s in quad_sigma])
    tau_array = np.asarray(tau_array, dtype=np.float64)

    psi = psi_k_vectorized(quad_sigma, tau_array, lambda_k_arr, q_k, eta, w_at_nodes)
    integrand = psi - 1.0  # [T, n_quad, d]
    log_ratio = np.einsum('tqd,q->td', integrand, quad_weights)  # [T, d]

    return sigma_T**2 * np.exp(2.0 * log_ratio)


# --- Shared-W analytical theory ---

def compute_A_k(lambda_k_arr, w_values, sigma_grid):
    """Shared-W convergence rates. Returns A_k [d], a_k_star [d], sigma_eff_sq."""
    lam = lambda_k_arr[:, None]
    sig2 = sigma_grid[None, :]**2
    w = w_values[None, :]

    A_k = (w * (lam + sig2)).mean(axis=1)
    B = w_values.mean()
    sigma_eff_sq = (w_values * sigma_grid**2).mean() / B
    a_k_star = lambda_k_arr * B / A_k

    return A_k, a_k_star, sigma_eff_sq


def compute_shared_w_variance(a_k, sigma_0=SIGMA_0, sigma_T=SIGMA_T):
    """lambda_tilde_k = sigma_T^2 * (sigma_0/sigma_T)^{2(1-a_k)}"""
    return sigma_T**2 * (sigma_0 / sigma_T) ** (2.0 * (1.0 - a_k))


# --- Shared-W theory under continuous log-normal sigma sampling (Binxu recipe) ---

def compute_sharedW_lognormal(lambda_k_arr, beta, P_mean=-1.2, P_std=1.2,
                              normalize=True):
    """Shared-W convergence rates for continuous LogNormal σ with w(σ) = σ^β.

    Uses the log-normal moment identity:
        E_{σ ~ LN(μ, s²)}[σ^k] = exp(k·μ + k²·s²/2)

    so E_LN[w(σ)] and E_LN[w(σ)·σ²] are analytically available for any β,
    without numerical integration or a discrete σ grid.

    Returns:
        dict with E_w, E_w_sigma2, lambda_crit, A_k [d], a_k_star [d].

        lambda_crit = E[w·σ²] / E[w] is the transition eigenvalue: modes with
        λ_k ≫ λ_crit are in the α≈1 spectral-bias regime; modes with
        λ_k ≪ λ_crit are saturated (τ_k* ≈ const, α ≈ 0).
    """
    P_std2 = P_std ** 2
    E_sig_b = math.exp(beta * P_mean + beta ** 2 * P_std2 / 2.0)
    E_sig_b2 = math.exp((beta + 2) * P_mean + (beta + 2) ** 2 * P_std2 / 2.0)

    if normalize:
        E_w = 1.0
        E_w_sigma2 = E_sig_b2 / E_sig_b
    else:
        E_w = E_sig_b
        E_w_sigma2 = E_sig_b2

    lambda_crit = E_w_sigma2 / E_w
    lam = np.asarray(lambda_k_arr, dtype=np.float64)
    A_k = lam * E_w + E_w_sigma2
    a_k_star = lam * E_w / A_k
    return dict(E_w=E_w, E_w_sigma2=E_w_sigma2, lambda_crit=lambda_crit,
                A_k=A_k, a_k_star=a_k_star)


def compute_sharedW_lognormal_trajectory(tau_array, lambda_k_arr, beta,
                                          P_mean=-1.2, P_std=1.2,
                                          sigma_0=SIGMA_0, sigma_T=SIGMA_T,
                                          eta=ETA, normalize=True):
    """[T, d] generated-variance trajectory under shared-W + LogNormal σ.

    Same functional form as compute_shared_w_trajectory, but A_k and a_k_star
    come from the analytic log-normal moments (compute_sharedW_lognormal)
    rather than a .mean() over a log-uniform σ grid.
    """
    out = compute_sharedW_lognormal(lambda_k_arr, beta, P_mean, P_std, normalize)
    A_k, a_k_star = out["A_k"], out["a_k_star"]
    tau_array = np.asarray(tau_array, dtype=np.float64)
    exponent = -2.0 * eta * tau_array[:, None] * A_k[None, :]
    a_k_traj = a_k_star[None, :] * (1.0 - np.exp(exponent))
    return sigma_T ** 2 * (sigma_0 / sigma_T) ** (2.0 * (1.0 - a_k_traj))


def compute_shared_w_trajectory(tau_array, lambda_k_arr, w_values, sigma_grid,
                                sigma_0=SIGMA_0, sigma_T=SIGMA_T, eta=ETA):
    """Shared-W generated-variance trajectory [T, d].

    Dynamics (gradient flow on a single W shared across sigma):
        a_k(tau) = a_k_star * (1 - exp(-2 * eta * A_k * tau))
        lambda_tilde_k(tau) = sigma_T^2 * (sigma_0/sigma_T)^{2(1 - a_k(tau))}

    where A_k = E_sigma[w(sigma)*(lambda_k + sigma^2)] and
          a_k_star = lambda_k * E_sigma[w] / A_k.

    Applicable to any denoiser with parameters shared across sigma (shared
    MLP included, in the convex / fixed-point limit).
    """
    A_k, a_k_star, _ = compute_A_k(lambda_k_arr, w_values, sigma_grid)
    tau_array = np.asarray(tau_array, dtype=np.float64)

    exponent = -2.0 * eta * tau_array[:, None] * A_k[None, :]  # [T, d]
    a_k_traj = a_k_star[None, :] * (1.0 - np.exp(exponent))    # [T, d]
    lambda_tilde = sigma_T ** 2 * (sigma_0 / sigma_T) ** (2.0 * (1.0 - a_k_traj))
    return lambda_tilde


# --- Emergence time computation ---

def compute_emergence_times(variance_traj, tau_array, lambda_k_arr,
                            threshold_frac=THRESHOLD_FRAC):
    """Fixed-fraction threshold crossing with linear interpolation. Returns [d]."""
    d = len(lambda_k_arr)
    tau_k_star = np.full(d, np.nan)
    thresholds = threshold_frac * lambda_k_arr

    for k in range(d):
        above = variance_traj[:, k] >= thresholds[k]
        if not np.any(above):
            continue
        idx = np.argmax(above)
        if idx == 0:
            tau_k_star[k] = tau_array[0]
        else:
            v0, v1 = variance_traj[idx - 1, k], variance_traj[idx, k]
            t0, t1 = tau_array[idx - 1], tau_array[idx]
            frac = (thresholds[k] - v0) / (v1 - v0) if v1 != v0 else 0.5
            tau_k_star[k] = t0 + frac * (t1 - t0)

    return tau_k_star


def compute_emergence_times_relative(variance_traj, tau_array, lambda_k_arr,
                                     threshold_frac=0.5, v_inf=None,
                                     min_range_frac=1e-3):
    """Relative-progress emergence time. Returns [d].

    progress_k(t) = (v_k(t) - v_k(0)) / (v_k(inf) - v_k(0))
    emergence_k   = first t where progress_k(t) >= threshold_frac

    Use this for *empirical* trajectories (ODE-sampled from a trained model)
    where the initial variance is determined by random-init sampling noise
    rather than zero. Under a fixed-fraction rule (compute_emergence_times)
    modes with small lambda_k are declared "emerged at step 0" because their
    threshold is tiny, which fools the detector. Relative progress removes
    that artifact by asking "when did each mode reach halfway between where
    it started and where it ended up?" — a fully data-driven definition.

    Args
    ----
    variance_traj : [T, d]  per-mode variance over checkpoints
    tau_array     : [T]     checkpoint indices (steps or continuous tau)
    lambda_k_arr  : [d]     data eigenvalues (used only for the min-range
                            filter)
    threshold_frac: float   progress level that counts as emergence (0.5 is
                            the halfway point)
    v_inf         : [d] or None. If None, empirical = variance_traj[-1].
                    Pass lambda_k_arr to use the theoretical asymptote
                    (more idealized but cleaner when training is converged).
    min_range_frac: float   if |v_inf - v_0| < min_range_frac * |lambda_k|,
                            the mode is treated as "did not move enough to
                            measure" and returns NaN. Guards against the
                            degenerate (flat) trajectory case.
    """
    variance_traj = np.asarray(variance_traj)
    T, d = variance_traj.shape
    tau_k_star = np.full(d, np.nan)

    v_0 = variance_traj[0]
    v_inf = variance_traj[-1] if v_inf is None else np.asarray(v_inf)
    denom = v_inf - v_0

    for k in range(d):
        if abs(denom[k]) < min_range_frac * abs(lambda_k_arr[k]):
            continue
        progress = (variance_traj[:, k] - v_0[k]) / denom[k]
        above = progress >= threshold_frac
        if not np.any(above):
            continue
        idx = np.argmax(above)
        if idx == 0:
            tau_k_star[k] = tau_array[0]
        else:
            p0, p1 = progress[idx - 1], progress[idx]
            t0, t1 = tau_array[idx - 1], tau_array[idx]
            frac = ((threshold_frac - p0) / (p1 - p0)) if p1 != p0 else 0.5
            tau_k_star[k] = t0 + frac * (t1 - t0)

    return tau_k_star


def compute_emergence_times_ak(a_k_traj, tau_array, a_k_star, convergence_frac=0.9):
    """a_k-based emergence: when a_k reaches convergence_frac * a_k*. Returns [d]."""
    d = a_k_traj.shape[1]
    tau_k_star = np.full(d, np.nan)
    thresholds = convergence_frac * a_k_star

    for k in range(d):
        if thresholds[k] <= 0 or not np.isfinite(thresholds[k]):
            continue
        above = a_k_traj[:, k] >= thresholds[k]
        if not np.any(above):
            continue
        idx = np.argmax(above)
        if idx == 0:
            tau_k_star[k] = tau_array[0]
        else:
            v0, v1 = a_k_traj[idx - 1, k], a_k_traj[idx, k]
            t0, t1 = tau_array[idx - 1], tau_array[idx]
            frac = (thresholds[k] - v0) / (v1 - v0) if v1 != v0 else 0.5
            tau_k_star[k] = t0 + frac * (t1 - t0)

    return tau_k_star


# --- Power-law fitting ---

def fit_power_law(tau_k_star, lambda_k_arr, method="ols"):
    """Fit tau_k* ~ lambda_k^{-alpha}. Returns dict with alpha, R2, n_used."""
    valid = np.isfinite(tau_k_star) & (tau_k_star > 0) & (lambda_k_arr > 0)
    n_used = int(np.sum(valid))

    if n_used < 3:
        return {"alpha": np.nan, "R2": np.nan, "n_used": n_used}

    log_lam = np.log(lambda_k_arr[valid]).reshape(-1, 1)
    log_tau = np.log(tau_k_star[valid])

    reg = LinearRegression().fit(log_lam, log_tau)
    return {
        "alpha": float(-reg.coef_[0]),
        "R2": float(reg.score(log_lam, log_tau)),
        "n_used": n_used,
    }
