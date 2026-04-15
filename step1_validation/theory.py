"""Analytical theory for the beta-emergence law.

Per-sigma ODE solutions, Phi integral via Gauss-Legendre quadrature,
shared-W ODE, emergence time computation, and power-law fitting.

Vectorized over eigenmodes AND tau values (no Python loops).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from sklearn.linear_model import LinearRegression

from .config import SIGMA_0, SIGMA_T, ETA, Q_K, THRESHOLD_FRAC, K_SIGMA


# ---------------------------------------------------------------------------
# Per-sigma ODE solution
# ---------------------------------------------------------------------------

def psi_k_vectorized(sigma_grid, tau, lambda_k_arr, q_k, eta, w_values):
    """Per-sigma ODE solution for all modes at all sigma values.

    Args:
        sigma_grid: [K] noise levels
        tau: scalar or [T] training times
        lambda_k_arr: [d] eigenvalues
        q_k: scalar initial overlap
        eta: learning rate
        w_values: [K] w(sigma) at sigma_grid

    Returns:
        If tau is scalar: [K, d]
        If tau is [T]: [T, K, d]
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    scalar_tau = (len(tau) == 1)

    K = len(sigma_grid)
    d = len(lambda_k_arr)

    sig2 = sigma_grid.reshape(K, 1) ** 2   # [K, 1]
    lam = lambda_k_arr.reshape(1, d)        # [1, d]
    wiener = lam / (lam + sig2)             # [K, d]

    w = w_values.reshape(K, 1)              # [K, 1]
    rate = w * (lam + sig2)                 # [K, d]

    # exponent: [T, K, d] = -2*eta * tau[T,1,1] * rate[1,K,d]
    exponent = -2.0 * eta * tau.reshape(-1, 1, 1) * rate.reshape(1, K, d)
    psi = wiener.reshape(1, K, d) + (q_k - wiener.reshape(1, K, d)) * np.exp(exponent)

    if scalar_tau:
        return psi[0]  # [K, d]
    return psi  # [T, K, d]


# ---------------------------------------------------------------------------
# Phi integral via Gauss-Legendre quadrature (fully vectorized)
# ---------------------------------------------------------------------------

def _setup_gauss_legendre(n_quad, sigma_0, sigma_T):
    """Pre-compute Gauss-Legendre nodes and weights in log-sigma space."""
    nodes, weights = leggauss(n_quad)
    a, b = np.log(sigma_0), np.log(sigma_T)
    log_sigma = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    quad_weights = 0.5 * (b - a) * weights
    sigma_values = np.exp(log_sigma)
    return sigma_values, quad_weights


def compute_phi_per_sigma(tau_array, lambda_k_arr, q_k=Q_K, eta=ETA,
                          w_fn=None, w_values=None, sigma_grid=None,
                          sigma_0=SIGMA_0, sigma_T=SIGMA_T, n_quad=100):
    """Compute generated variance trajectories via per-sigma Phi integral.

    Fully vectorized: no Python loop over tau values.
    Memory: O(T * n_quad * d). For T=500, n_quad=100, d=200: ~80MB.

    Args:
        tau_array: [T] training times
        lambda_k_arr: [d] eigenvalues
        q_k: initial overlap (0 for W=0 init)
        eta: learning rate
        w_fn: callable w(sigma) OR
        w_values, sigma_grid: pre-evaluated weights
        sigma_0, sigma_T: noise bounds
        n_quad: quadrature points

    Returns:
        variance_traj: [T, d] generated variance lambda_tilde_k(tau)
    """
    quad_sigma, quad_weights = _setup_gauss_legendre(n_quad, sigma_0, sigma_T)

    if w_fn is not None:
        w_at_nodes = np.array([w_fn(s) for s in quad_sigma])
    elif w_values is not None and sigma_grid is not None:
        w_at_nodes = np.interp(np.log(quad_sigma), np.log(sigma_grid), w_values)
    else:
        raise ValueError("Provide either w_fn or (w_values, sigma_grid)")

    tau_array = np.asarray(tau_array, dtype=np.float64)

    # psi: [T, K, d] — all tau values at once
    psi = psi_k_vectorized(quad_sigma, tau_array, lambda_k_arr, q_k, eta, w_at_nodes)
    # psi shape: [T, n_quad, d]

    integrand = psi - 1.0  # [T, n_quad, d]

    # Quadrature sum: contract over n_quad dimension
    # log(Phi_0/Phi_T) = int_{ln(sigma_0)}^{ln(sigma_T)} (psi-1) du
    log_ratio = np.einsum('tqd,q->td', integrand, quad_weights)  # [T, d]

    # lambda_tilde_k = sigma_T^2 * (Phi_0/Phi_T)^2 = sigma_T^2 * exp(2 * log_ratio)
    variance_traj = sigma_T**2 * np.exp(2.0 * log_ratio)

    return variance_traj


# ---------------------------------------------------------------------------
# Shared-W model analytical theory
# ---------------------------------------------------------------------------

def compute_A_k(lambda_k_arr, w_values, sigma_grid):
    """Shared-W convergence rates and fixed points.

    Returns:
        A_k: [d] convergence rates
        a_k_star: [d] fixed points (compromise Wiener filter)
        sigma_eff_sq: scalar effective noise variance
    """
    lam = lambda_k_arr[:, None]    # [d, 1]
    sig2 = sigma_grid[None, :]**2  # [1, K]
    w = w_values[None, :]          # [1, K]

    A_k = (w * (lam + sig2)).mean(axis=1)  # [d]
    B = w_values.mean()
    C = (w_values * sigma_grid**2).mean()
    sigma_eff_sq = C / B

    a_k_star = lambda_k_arr * B / A_k  # [d]

    return A_k, a_k_star, sigma_eff_sq


def compute_shared_w_trajectory(tau_array, lambda_k_arr, A_k, a_k_star, a_k_init=0.0):
    """Shared-W analytical trajectory: a_k(tau) = a_k* + (a_k0 - a_k*) * exp(-2*eta*tau*A_k).

    Returns:
        a_k_traj: [T, d]
    """
    tau = np.asarray(tau_array)[:, None]  # [T, 1]
    return a_k_star[None, :] + (a_k_init - a_k_star[None, :]) * np.exp(-2.0 * ETA * tau * A_k[None, :])


def compute_shared_w_variance(a_k, sigma_0=SIGMA_0, sigma_T=SIGMA_T):
    """Generated variance from shared-W model (closed form).

    lambda_tilde_k = sigma_T^2 * (sigma_0/sigma_T)^{2(1-a_k)}
    """
    ratio = sigma_0 / sigma_T
    return sigma_T**2 * ratio ** (2.0 * (1.0 - a_k))


def compute_inaccessible_modes(lambda_k_arr, a_k_star, sigma_0=SIGMA_0,
                                sigma_T=SIGMA_T, threshold_frac=THRESHOLD_FRAC):
    """Identify modes where lambda_tilde_k(inf) < threshold * lambda_k.

    Returns:
        mask: [d] boolean (True = inaccessible)
        lambda_tilde_inf: [d]
    """
    lambda_tilde_inf = compute_shared_w_variance(a_k_star, sigma_0, sigma_T)
    mask = lambda_tilde_inf < threshold_frac * lambda_k_arr
    return mask, lambda_tilde_inf


# ---------------------------------------------------------------------------
# Emergence time computation
# ---------------------------------------------------------------------------

def compute_emergence_times(variance_traj, tau_array, lambda_k_arr,
                            threshold_frac=THRESHOLD_FRAC):
    """Find emergence times via fixed-fraction threshold crossing.

    tau_k* = first time where lambda_tilde_k(tau) >= threshold_frac * lambda_k
    Linear interpolation between checkpoints.

    Returns:
        tau_k_star: [d] (NaN for non-emerged modes)
    """
    d = len(lambda_k_arr)
    tau_k_star = np.full(d, np.nan)
    thresholds = threshold_frac * lambda_k_arr

    for k in range(d):
        thr = thresholds[k]
        var_k = variance_traj[:, k]

        above = var_k >= thr
        if not np.any(above):
            continue

        idx = np.argmax(above)
        if idx == 0:
            tau_k_star[k] = tau_array[0]
        else:
            v0, v1 = var_k[idx - 1], var_k[idx]
            t0, t1 = tau_array[idx - 1], tau_array[idx]
            frac = (thr - v0) / (v1 - v0) if v1 != v0 else 0.5
            tau_k_star[k] = t0 + frac * (t1 - t0)

    return tau_k_star


def compute_emergence_times_ak(a_k_traj, tau_array, a_k_star, convergence_frac=0.9):
    """Find emergence times based on a_k reaching a fraction of a_k*.

    Measures HOW FAST each mode converges, independent of the sampling ODE.
    This is the appropriate metric for the shared-W model, where the sampling
    ODE is degenerate (sigma_eff^2 >> lambda_max makes all modes inaccessible).

    tau_k* = first time where a_k(tau) >= convergence_frac * a_k*

    Args:
        a_k_traj: [T, d] trajectory of eigenmode projections
        tau_array: [T] training times
        a_k_star: [d] fixed points
        convergence_frac: fraction of a_k* (default 0.9)

    Returns:
        tau_k_star: [d] (NaN for non-converged modes)
    """
    d = a_k_traj.shape[1]
    tau_k_star = np.full(d, np.nan)
    thresholds = convergence_frac * a_k_star

    for k in range(d):
        thr = thresholds[k]
        if thr <= 0 or not np.isfinite(thr):
            continue

        ak = a_k_traj[:, k]
        above = ak >= thr
        if not np.any(above):
            continue

        idx = np.argmax(above)
        if idx == 0:
            tau_k_star[k] = tau_array[0]
        else:
            v0, v1 = ak[idx - 1], ak[idx]
            t0, t1 = tau_array[idx - 1], tau_array[idx]
            frac = (thr - v0) / (v1 - v0) if v1 != v0 else 0.5
            tau_k_star[k] = t0 + frac * (t1 - t0)

    return tau_k_star


def compute_emergence_times_ak_analytical(A_k, convergence_frac=0.9):
    """Analytical emergence times for the shared-W model.

    tau_k* = -ln(1 - convergence_frac) / (2 * eta * A_k)

    This is exact for the continuous gradient flow ODE.

    Returns:
        tau_k_star: [d]
    """
    return -np.log(1.0 - convergence_frac) / (2.0 * ETA * A_k)


# ---------------------------------------------------------------------------
# Power-law fitting
# ---------------------------------------------------------------------------

def fit_power_law(tau_k_star, lambda_k_arr, method="ols"):
    """Fit tau_k* = A * lambda_k^{-alpha} via log-log regression.

    Returns:
        dict with keys: alpha, R2, n_used, intercept
        alpha is POSITIVE when larger eigenvalues emerge faster
    """
    valid = np.isfinite(tau_k_star) & (tau_k_star > 0) & (lambda_k_arr > 0)
    n_used = int(np.sum(valid))

    if n_used < 3:
        return {"alpha": np.nan, "R2": np.nan, "n_used": n_used, "intercept": np.nan}

    log_lam = np.log(lambda_k_arr[valid])
    log_tau = np.log(tau_k_star[valid])

    if method == "ols":
        reg = LinearRegression()
        reg.fit(log_lam.reshape(-1, 1), log_tau)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        R2 = reg.score(log_lam.reshape(-1, 1), log_tau)
    elif method == "theil-sen":
        from sklearn.linear_model import TheilSenRegressor
        reg = TheilSenRegressor(random_state=0)
        reg.fit(log_lam.reshape(-1, 1), log_tau)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        ss_res = np.sum((log_tau - reg.predict(log_lam.reshape(-1, 1)))**2)
        ss_tot = np.sum((log_tau - log_tau.mean())**2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    alpha = -slope

    return {"alpha": float(alpha), "R2": float(R2), "n_used": n_used,
            "intercept": float(intercept)}
