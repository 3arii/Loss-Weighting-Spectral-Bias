import numpy as np
from scipy.stats import norm as scipy_norm


def lognormal_pdf(sigma, P_mean=-1.2, P_std=1.2):
    s = np.asarray(sigma, dtype=float)
    return np.exp(-0.5 * ((np.log(s) - P_mean) / P_std) ** 2) / (
        s * P_std * np.sqrt(2 * np.pi)
    )


def _edm_loss_weight(sigma, sigma_data=0.5):
    s = float(sigma)
    return (s ** 2 + sigma_data ** 2) / (s * sigma_data) ** 2


def _min_snr_weight(sigma, sigma_data=0.5, gamma=5):
    snr = sigma_data ** 2 / sigma ** 2
    return min(snr, gamma) / snr


def _p2_weight(sigma, sigma_data=0.5, gamma=1, k=1):
    snr = sigma_data ** 2 / sigma ** 2
    return 1.0 / (snr * (k + snr) ** gamma)


def diffusionflow_edm_weight(log_snr):
    return scipy_norm.pdf(log_snr, loc=2.4, scale=2.4) * (
        np.exp(-log_snr) + 0.5 ** 2
    )


def _uniform(**kwargs):
    return (lambda sigma: 1, lambda sigma: 1)


def _edm(sigma_data=0.5, P_mean=-1.2, P_std=1.2, **kwargs):
    def eta_fn(sigma):
        pdf = lognormal_pdf(sigma, P_mean=P_mean, P_std=P_std)
        weight = _edm_loss_weight(sigma, sigma_data=sigma_data)
        return float(pdf * weight)
    return (eta_fn, lambda sigma: 1)


def _min_snr_gamma(sigma_data=0.5, gamma=5, **kwargs):
    return (
        lambda sigma: _min_snr_weight(sigma, sigma_data, gamma),
        lambda sigma: 1,
    )


def _inverse_variance(**kwargs):
    return (lambda sigma: 1.0 / sigma ** 2, lambda sigma: 1)


def _p2(sigma_data=0.5, gamma=1, k=1, **kwargs):
    return (
        lambda sigma: _p2_weight(sigma, sigma_data, gamma, k),
        lambda sigma: 1,
    )


def _lognormal_only(P_mean=-1.2, P_std=1.2, **kwargs):
    return (
        lambda sigma: float(lognormal_pdf(sigma, P_mean, P_std)),
        lambda sigma: 1,
    )


def _power_law(alpha=-1, **kwargs):
    return (lambda sigma: sigma ** alpha, lambda sigma: 1)


WEIGHTINGS = {
    "uniform": _uniform,
    "edm": _edm,
    "min-snr-gamma": _min_snr_gamma,
    "inverse-variance": _inverse_variance,
    "p2": _p2,
    "lognormal-only": _lognormal_only,
    "power-law": _power_law,
}


def get_weighting(name, sigma_data=0.5, **kwargs):
    if name not in WEIGHTINGS:
        raise ValueError(
            f"Unknown weighting '{name}'. Available: {list(WEIGHTINGS.keys())}"
        )
    return WEIGHTINGS[name](sigma_data=sigma_data, **kwargs)


def list_weightings():
    return list(WEIGHTINGS.keys())
