# -----------------------------------------------------------------------------
# ARGUS generators - Ratio-of-Uniforms method based on Gamma distribution
# -----------------------------------------------------------------------------

import numpy as np
from numpy.random import default_rng
from _utils import _lazywhere


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms without mode shift
# -----------------------------------------------------------------------------

def _get_rect_rou_gamma(chi):
    def _sqrt_f(x):
        return x**0.25 * np.exp(-x/2)

    chi2 = chi * chi
    m = min(chi2 / 2, 0.5)

    if chi2 >= 5:
        vmax = 2.5 * _sqrt_f(2.5)
    else:
        vmax = chi2 / 2 * _sqrt_f(chi2 / 2)
    umax = _sqrt_f(m)
    vmin = 0

    return umax, vmin, vmax


def rvs_rou_gamma(chi, size=1, seed=None):
    def qpdf(x, chi):
        return _lazywhere((x >= 0) & (x <= chi**2 / 2), (x, chi),
                          lambda x, chi: np.sqrt(x) * np.exp(-x), 0)

    umax, vmin, vmax = _get_rect_rou_gamma(chi)

    y = np.zeros(size)
    rg = default_rng(seed)
    simulated = 0
    while simulated < size:
        k = size - simulated
        u1 = umax * rg.uniform(size=k)
        v1 = rg.uniform(vmin, vmax, size=k)
        rvs = v1 / u1
        accept = (u1**2 <= qpdf(rvs, chi))
        num_accept = np.sum(accept)
        if num_accept > 0:
            y[simulated:(simulated + num_accept)] = rvs[accept]
            simulated += num_accept

    return np.sqrt(1 - 2*y/chi**2)


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms with mode shift
# -----------------------------------------------------------------------------

def _get_rect_rou_shifted_gamma(chi):
    def _sqrt_f(x):
        return x**0.25 * np.exp(-x/2)

    chi2 = chi * chi
    if chi >= 1:
        m = 0.5
    else:
        m = chi2 / 2

    if chi >= 1:
        x_max = min(1.5 + np.sqrt(2), chi2 / 2)
        x_min = 1.5 - np.sqrt(2)
        vmax = (x_max - m) * _sqrt_f(x_max)
    else:
        vmax = 0
        x_min = m/2 - np.sqrt(4*m**2 + 12*m + 25)/4 + 5/4

    umax = _sqrt_f(m)
    vmin = (x_min - m) * _sqrt_f(x_min)

    return umax, vmin, vmax


def rvs_rou_shifted_gamma(chi, size=1, seed=None):
    def qpdf(x, chi):
        return _lazywhere((x >= 0) & (x <= chi**2 / 2), (x, chi),
                          lambda x, chi: np.exp(0.5*np.log(x) - x), 0)

    chi2 = chi * chi
    if chi >= 1:
        m = 0.5
    else:
        m = chi2 / 2
    umax, vmin, vmax = _get_rect_rou_shifted_gamma(chi)
    
    # y = stats.rvs_ratio_uniforms(lambda x: qpdf(x, chi), umax, vmin, vmax, size=size, c=m)
    y = np.zeros(size)
    rg = default_rng(seed)
    simulated = 0
    while simulated < size:
        k = size - simulated
        u1 = umax * rg.uniform(size=k)
        v1 = rg.uniform(vmin, vmax, size=k)
        rvs = v1 / u1 + m
        accept = (u1**2 <= qpdf(rvs, chi))
        num_accept = np.sum(accept)
        if num_accept > 0:
            y[simulated:(simulated + num_accept)] = rvs[accept]
            simulated += num_accept
    return np.sqrt(1 - 2*y/chi2)
