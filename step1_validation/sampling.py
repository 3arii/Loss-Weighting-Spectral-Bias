"""ODE sampling + generated-variance evaluation for MLP denoisers.

Uses the EDM probability-flow ODE with an x_0-prediction denoiser:
    score(x, sigma) = (D(x, sigma) - x) / sigma^2
    dx/dsigma      = -sigma * score = (x - D(x, sigma)) / sigma

Heun integration from sigma_T down to sigma_0. For diagonal-Gaussian data
(variance lambda_k per coordinate), the generated per-eigenmode variance
is just the per-coordinate sample variance.
"""

import numpy as np
import torch


@torch.no_grad()
def heun_sample(model, n_samples, ndim, sigma_schedule, device):
    """EDM Heun sampler. sigma_schedule: 1-D tensor descending from sigma_T
    to sigma_0 (> 0). Returns clean samples [n_samples, ndim]."""
    sigma_schedule = sigma_schedule.to(device)
    x = torch.randn(n_samples, ndim, device=device) * sigma_schedule[0]

    for i in range(len(sigma_schedule) - 1):
        s_cur = sigma_schedule[i]
        s_next = sigma_schedule[i + 1]

        sig_cur = s_cur.expand(n_samples)
        D_cur = model(x, sig_cur)
        d_cur = (x - D_cur) / s_cur
        x_next = x + (s_next - s_cur) * d_cur

        if s_next > 0:
            sig_next = s_next.expand(n_samples)
            D_next = model(x_next, sig_next)
            d_next = (x_next - D_next) / s_next
            x = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
        else:
            x = x_next

    return x


def make_sigma_schedule(sigma_min, sigma_max, num_steps, rho=7.0):
    """EDM Karras schedule (log-spaced with rho warping)."""
    ramp = torch.linspace(0, 1, num_steps)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


@torch.no_grad()
def generated_variance_per_mode(model, n_samples, ndim, sigma_min, sigma_max,
                                num_ode_steps, device, rho=7.0):
    """Sample via ODE, return per-coordinate variance [ndim].

    For diagonal-Gaussian data, coordinate variances = eigenmode variances.
    """
    schedule = make_sigma_schedule(sigma_min, sigma_max, num_ode_steps, rho=rho)
    samples = heun_sample(model, n_samples, ndim, schedule, device)
    return samples.var(dim=0, unbiased=False).cpu().numpy()
