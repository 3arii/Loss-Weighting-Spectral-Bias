# Diagnosis: NaN at β=2.5 in per-sigma Φ integral

**Date:** 2026-04-14
**Script:** `step1_validation/run_per_sigma.py`
**Config:** σ_min=0.002, σ_max=80, η=1.0, d=200, τ ∈ [1e-2, 1e6], threshold_frac=0.5

---

## Summary

`alpha_phi=nan, R2=nan` at β=2.5 is **not a numerical bug** in the integral.
It is a **domain validity failure**: at β=2.5, only 1 out of 200 modes emerges
within the training time window τ ∈ [1e-2, 1e6], so `fit_power_law` correctly
returns NaN (requires n_used ≥ 3).

---

## Pre-check — Weight magnitude at σ_min and σ_max

Before tracing the code, evaluate how large w(σ) = σ^β gets across the σ range:

```python
sigma_min, sigma_max = 0.002, 80.0
beta = 2.5

w(sigma_min) = 0.002^2.5 = 1.79e-7
w(sigma_max) = 80.0^2.5  = 5.72e+4
ratio         = (80/0.002)^2.5 = 3.20e+11   (11.5 decades)
```

| σ | w(σ) = σ^2.5 |
|---|---|
| 0.002 (σ_min) | 1.79e-7 |
| 80.0 (σ_max) | **5.72e+4** |
| ratio max/min | **3.20e+11** (11.5 decades) |

The weight function spans **11.5 orders of magnitude** across the σ range.
This extreme dynamic range is the upstream cause of all downstream issues.

---

## Check 1 — Exponent overflow in `psi_k_vectorized`

The per-sigma ODE exponent is:

```
exp(-2η · w(σ) · τ · (λ_k + σ²))
```

At σ_max=80, β=2.5:

| σ | w(σ) = σ^β | rate = w·(λ₀+σ²) | exponent at τ=1 |
|---|---|---|---|
| 0.002 | 1.79e-7 | 1.80e-7 | -3.6e-7 (ok) |
| 80.0 | 5.72e+4 | 3.64e+8 | **-7.28e+8** (underflow) |

`np.exp(-7e8) = 0.0` — underflows to zero **silently**, not NaN.
**No NaN is introduced here.** ✓

---

## Check 2 — Gauss-Legendre quadrature sum

```
log_ratio = ∫_{ln σ_min}^{ln σ_max} (ψ − 1) d(ln σ)
variance  = σ_T² · exp(2 · log_ratio)
```

Full sweep over all (T=500, K=100, d=200) entries:

| Quantity | Min | Max | NaN? | Inf? |
|---|---|---|---|---|
| exponent | -7.28e+14 | -1.80e-11 | No | No |
| entries underflowing (<-745) | 4,255,022 / 10,000,000 | — | — | — |
| log_ratio | -10.60 | -4.69 | **No** | **No** |
| variance_traj | 4.0e-6 | 0.54 | **No** | **No** |

The integral is well-behaved. **No NaN introduced here.** ✓

---

## Check 3 — Root cause: emergence-time fitting

At β=2.5, the weights are enormous at large σ but tiny at small σ.
The dynamics are dominated by the slow end (σ_min), where:

```
rate(σ_min=0.002) = w(0.002) · (λ + σ²) ≈ 1.8e-7
```

Even for the **largest** mode (λ₀=1, threshold=0.5), convergence only barely
occurs by τ ≈ 8.8e5. All 199 smaller modes never cross their thresholds within
τ_max=1e6:

```
emerged: 1 / 200
tau_k[0] = 884,831    (barely within window)
tau_k[1:] = NaN       (never emerged)
```

`fit_power_law` requires n_used ≥ 3:

```python
if n_used < 3:
    return {"alpha": np.nan, "R2": np.nan, ...}   # theory.py line 287
```

With n_used=1, this guard triggers → **alpha_phi=nan, R2=nan**.

---

## Full results table (d=200, summary, no emergence arrays)

| alpha_data | β | alpha_phi | R² | n_used |
|---|---|---|---|---|
| 0.56 | -2.5 | -0.250 | 1.000 | 200 |
| 0.56 | -2.0 | -0.000 | 0.969 | 200 |
| 0.56 | -1.5 | 0.250 | 1.000 | 200 |
| 0.56 | -1.0 | 0.500 | 1.000 | 200 |
| 0.56 | -0.5 | 0.750 | 1.000 | 200 |
| 0.56 | 0.0 | 0.955 | 1.000 | 200 |
| 0.56 | 0.5 | 0.998 | 1.000 | 200 |
| 0.56 | 1.0 | 1.000 | 1.000 | 200 |
| 0.56 | 1.5 | 1.000 | 1.000 | 200 |
| 0.56 | 2.0 | 1.000 | 1.000 | 200 |
| 0.56 | **2.5** | **NaN** | **NaN** | **1** |
| 1.0 | -2.5 | -0.250 | 1.000 | 200 |
| 1.0 | -2.0 | -0.000 | 0.797 | 200 |
| 1.0 | -1.5 | 0.250 | 1.000 | 200 |
| 1.0 | -1.0 | 0.500 | 1.000 | 200 |
| 1.0 | -0.5 | 0.749 | 1.000 | 200 |
| 1.0 | 0.0 | 0.941 | 1.000 | 200 |
| 1.0 | 0.5 | 0.993 | 1.000 | 200 |
| 1.0 | 1.0 | 0.998 | 1.000 | 200 |
| 1.0 | 1.5 | 0.998 | 1.000 | 200 |
| 1.0 | 2.0 | 1.000 | 1.000 | 19 |
| 1.0 | **2.5** | **NaN** | **NaN** | **1** |
| 2.0 | -2.5 | -0.284 | 0.990 | 200 |
| 2.0 | -2.0 | -0.029 | 0.565 | 200 |
| 2.0 | -1.5 | 0.219 | 0.980 | 200 |
| 2.0 | -1.0 | 0.451 | 0.988 | 200 |
| 2.0 | -0.5 | 0.648 | 0.984 | 200 |
| 2.0 | 0.0 | 0.779 | 0.981 | 200 |
| 2.0 | 0.5 | 0.836 | 0.982 | 200 |
| 2.0 | 1.0 | 0.968 | 0.999 | 82 |
| 2.0 | 1.5 | 0.998 | 1.000 | 18 |
| 2.0 | 2.0 | 1.000 | 1.000 | 4 |
| 2.0 | **2.5** | **NaN** | **NaN** | **1** |

Note: n_used drops sharply for large β + large alpha_data — heavier-tailed data
combined with large-β weights makes most modes too slow to emerge within τ_max=1e6.

---

## Physical interpretation

At β=2.5, w(σ) = σ^2.5 places overwhelming weight on large noise levels.
The learning dynamics at large σ converge instantly (exponent ~ -7e8 at τ=1),
but the threshold metric `λ̃_k ≥ 0.5 λ_k` is evaluated on the **full Φ integral**,
which is dominated by the slow small-σ regime. Since small-σ rates are ~1e-7,
spectral emergence in the Φ sense is extremely slow — slower than τ_max=1e6.

**β=2.5 is outside the observable emergence regime for σ ∈ [0.002, 80], τ ≤ 1e6.**

### Suggested fixes

1. **Extend τ_max** to ~1e10 (but runtime scales accordingly)
2. **Cap σ_max** at a lower value (e.g. 10) to reduce the weight dynamic range
3. **Treat β ≥ 2.5 as out-of-domain** and skip emergence-law fitting for those cases

---

*Full JSON results:* `$STORE_DIR/step1_results/per_sigma/per_sigma_analytic_d200.json`
