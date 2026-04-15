"""Fast local test — validates all modules work end-to-end.

Runs in <10s on CPU with tiny parameters (d=5, N=50, 100 steps).
Tests: imports, theory math, loss computation, training loop, JSON output.

Usage:
    python -m step1_validation.test_local
"""

import sys
import json
import tempfile
import os
import numpy as np
import torch

# -- Test 1: Imports ----------------------------------------------------------
print("Test 1: Imports...", end=" ")
from .config import (SIGMA_0, SIGMA_T, ETA, Q_K, THRESHOLD_FRAC,
                     get_sigma_grid, get_sigma_grid_np, get_checkpoint_steps,
                     make_eigenvalues)
from .models import LinearDenoiserShared
from .losses import DeterministicPowerLawLoss, GeneralWeightingLoss
from .theory import (
    psi_k_vectorized, compute_phi_per_sigma, compute_A_k,
    compute_shared_w_trajectory, compute_shared_w_variance,
    compute_inaccessible_modes, compute_emergence_times,
    compute_emergence_times_ak, compute_emergence_times_ak_analytical,
    fit_power_law,
)
print("OK")

# -- Test 2: Config helpers ---------------------------------------------------
print("Test 2: Config helpers...", end=" ")
sig_grid = get_sigma_grid(10)
assert sig_grid.shape == (10,), f"Expected (10,), got {sig_grid.shape}"
sig_np = get_sigma_grid_np(10)
assert sig_np.shape == (10,), f"Expected (10,), got {sig_np.shape}"
ckpts = get_checkpoint_steps(100, 20)
assert ckpts[0] == 0 and ckpts[-1] == 99
eigs = make_eigenvalues(1.0, 5)
assert eigs[0] == 1.0 and len(eigs) == 5
print("OK")

# -- Test 3: psi_k boundary conditions ---------------------------------------
print("Test 3: psi_k boundaries...", end=" ")
d, K = 5, 10
lam = make_eigenvalues(1.0, d)
sig = get_sigma_grid_np(K)
w = np.ones(K)

# tau=0 => psi = q_k
psi_0 = psi_k_vectorized(sig, 0.0, lam, q_k=0.0, eta=1.0, w_values=w)
assert psi_0.shape == (K, d)
assert np.allclose(psi_0, 0.0, atol=1e-10), f"psi(tau=0) should be q_k=0, got max={psi_0.max()}"

# tau=very large => psi = Wiener filter
psi_inf = psi_k_vectorized(sig, 1e10, lam, q_k=0.0, eta=1.0, w_values=w)
wiener = lam[None, :] / (lam[None, :] + sig[:, None]**2)
assert np.allclose(psi_inf, wiener, atol=1e-6), "psi(tau=inf) should be Wiener filter"

# tau as array => [T, K, d]
psi_arr = psi_k_vectorized(sig, np.array([0.0, 1e10]), lam, q_k=0.0, eta=1.0, w_values=w)
assert psi_arr.shape == (2, K, d), f"Expected (2, {K}, {d}), got {psi_arr.shape}"
print("OK")

# -- Test 4: Phi integral produces reasonable variance ------------------------
print("Test 4: Phi integral...", end=" ")
tau_arr = np.geomspace(0.1, 1e4, 50)
var_traj = compute_phi_per_sigma(tau_arr, lam, q_k=0.0, eta=1.0,
                                  w_fn=lambda s: 1.0, n_quad=50)
assert var_traj.shape == (50, d)
# At tau=0.1 (early): variance should be small
assert var_traj[0, 0] < lam[0], "Early variance should be below target"
# At tau=1e4 (late): largest mode should have emerged
assert var_traj[-1, 0] > 0.5 * lam[0], f"Late var[0]={var_traj[-1,0]:.4f} should exceed 0.5*lam[0]={0.5*lam[0]:.4f}"
print("OK")

# -- Test 5: Emergence times + power law fit ----------------------------------
print("Test 5: Emergence times...", end=" ")
tau_star = compute_emergence_times(var_traj, tau_arr, lam, threshold_frac=0.5)
n_emerged = np.sum(np.isfinite(tau_star))
assert n_emerged >= 1, "At least mode 0 should emerge"
# Mode 0 (largest eigenvalue) should emerge first
if n_emerged >= 2:
    assert tau_star[0] < tau_star[1], "Largest eigenvalue should emerge first"

fit = fit_power_law(tau_star, lam)
if n_emerged >= 3:
    assert np.isfinite(fit["alpha"]), "alpha should be finite"
    assert fit["R2"] > 0.5, f"R2={fit['R2']:.3f} too low for clean power-law data"
    # For uniform weighting, alpha should be near 1.0
    assert 0.5 < fit["alpha"] < 1.5, f"alpha={fit['alpha']:.3f} not near 1.0 for uniform weighting"
print(f"OK (emerged={n_emerged}/{d}, alpha={fit['alpha']:.3f}, R2={fit['R2']:.3f})")

# -- Test 6: Shared-W theory -------------------------------------------------
print("Test 6: Shared-W theory...", end=" ")
A_k, a_k_star, sigma_eff_sq = compute_A_k(lam, w, sig)
assert A_k.shape == (d,)
assert a_k_star.shape == (d,)
assert sigma_eff_sq > 0
# a_k_star should be positive and < 1
assert np.all(a_k_star > 0) and np.all(a_k_star < 1)

# Shared-W variance at boundaries
var_0 = compute_shared_w_variance(0.0)
assert np.isclose(var_0, SIGMA_0**2, rtol=0.01), f"var(a_k=0)={var_0:.2e} != sigma_0^2={SIGMA_0**2:.2e}"
var_1 = compute_shared_w_variance(1.0)
assert np.isclose(var_1, SIGMA_T**2, rtol=0.01), f"var(a_k=1)={var_1:.2e} != sigma_T^2={SIGMA_T**2:.2e}"

# Inaccessible modes
mask, _ = compute_inaccessible_modes(lam, a_k_star)
assert mask.shape == (d,)

# a_k-based emergence (analytical)
tau_ak_an = compute_emergence_times_ak_analytical(A_k, convergence_frac=0.9)
assert np.all(tau_ak_an > 0) and np.all(np.isfinite(tau_ak_an))
fit_ak = fit_power_law(tau_ak_an, lam)
assert np.isfinite(fit_ak["alpha"]), "a_k analytical alpha should be finite"
print(f"OK (sigma_eff^2={sigma_eff_sq:.1f}, inacc={mask.sum()}/{d}, "
      f"alpha_ak={fit_ak['alpha']:.3f})")

# -- Test 7: Model forward pass ----------------------------------------------
print("Test 7: Model...", end=" ")
model = LinearDenoiserShared(d)
x = torch.randn(4, d)
out = model(x, sigma=torch.tensor(1.0))
assert out.shape == (4, d)
# W=0, b=0 => output should be zero
assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)
print("OK")

# -- Test 8: Loss computation ------------------------------------------------
print("Test 8: Loss...", end=" ")
loss_fn = DeterministicPowerLawLoss(beta=0.0, K_sigma=10, lambda_max=1.0)
loss = loss_fn(model, x)
assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
assert loss.item() > 0, "Loss should be positive"
assert torch.isfinite(loss), "Loss should be finite"
# Weight normalization: check normalization_factor is finite
assert np.isfinite(loss_fn.normalization_factor)
print(f"OK (loss={loss.item():.4f}, norm_factor={loss_fn.normalization_factor:.4f})")

# -- Test 9: Training loop (tiny) --------------------------------------------
print("Test 9: Training (50 steps)...", end=" ")
model = LinearDenoiserShared(d)
loss_fn = DeterministicPowerLawLoss(beta=0.0, K_sigma=10, lambda_max=1.0)
X = torch.randn(50, d) * torch.tensor(np.sqrt(lam), dtype=torch.float32)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []
for step in range(50):
    loss = loss_fn(model, X)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    losses.append(loss.item())

assert all(np.isfinite(losses)), "All losses should be finite (no divergence)"
assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
a_k_final = model.W.data.diag().numpy()
assert np.all(np.isfinite(a_k_final)), "W diagonal should be finite"
print(f"OK (loss: {losses[0]:.4f} -> {losses[-1]:.4f}, a_k[0]={a_k_final[0]:.6f})")

# -- Test 10: a_k emergence from training trajectory -------------------------
print("Test 10: a_k emergence...", end=" ")
# Build a fake a_k trajectory that converges exponentially
A_k_norm = A_k / A_k.max()
a_k_star_norm = lam * (w.mean() / A_k.max()) / A_k_norm
tau_arr_train = np.arange(0, 1000, dtype=float)
a_k_traj = compute_shared_w_trajectory(tau_arr_train, lam, A_k_norm, a_k_star_norm)
assert a_k_traj.shape == (1000, d)

tau_ak = compute_emergence_times_ak(a_k_traj, tau_arr_train, a_k_star_norm, 0.9)
n_emerged_ak = np.sum(np.isfinite(tau_ak))
fit_ak_traj = fit_power_law(tau_ak, lam)
print(f"OK (emerged={n_emerged_ak}/{d}, alpha={fit_ak_traj['alpha']:.3f})")

# -- Test 11: JSON serialization ---------------------------------------------
print("Test 11: JSON output...", end=" ")
result = {
    "alpha": float(fit["alpha"]),
    "eigenvalues": lam.tolist(),
    "emergence_times": np.where(np.isnan(tau_star), None, tau_star).tolist(),
    "a_k_trajectories": a_k_traj[:5].tolist(),
}
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(result, f, indent=2, default=str)
    tmppath = f.name
with open(tmppath) as f:
    loaded = json.load(f)
os.unlink(tmppath)
assert loaded["alpha"] == result["alpha"]
print("OK")

# -- Summary ------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL 11 TESTS PASSED")
print("=" * 60)
