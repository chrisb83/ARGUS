# -----------------------------------------------------------------------------
# implementation of ARGUS generators based on transformation of Maxwell rvs
# -----------------------------------------------------------------------------
import numpy as np
from numpy.random import default_rng
from _utils import _lazywhere


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms without mode shift
# -----------------------------------------------------------------------------

def rvs_rou_maxwell(chi, size=1, seed=None):
    def f(x):
        return _lazywhere((x >= 0) & (x <= chi), (x, chi),
                          lambda x, chi: np.exp(2*np.log(x) - x**2/2), 0)

    if chi < np.sqrt(2):
        umax = np.sqrt(f(chi))
    else:
        umax = np.sqrt(2) / np.exp(0.5)

    if chi < 2:
        vmax = chi * np.sqrt(f(chi))
    else:
        vmax = 4 / np.exp(1)

    y = np.zeros(size)
    rg = default_rng(seed)
    simulated = 0
    while simulated < size:
        k = size - simulated
        u1 = umax * rg.uniform(size=k)
        v1 = rg.uniform(0.0, vmax, size=k)
        rvs = v1 / u1
        accept = (u1**2 <= f(rvs))
        num_accept = np.sum(accept)
        if num_accept > 0:
            y[simulated:(simulated + num_accept)] = rvs[accept]
            simulated += num_accept

    return np.sqrt(1 - y*y / chi**2)


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms with mode shift
# -----------------------------------------------------------------------------
def _get_rectangle_rou_shifted_maxwell(chi):
    def _sqrt_f(u):
        return u * np.exp(-u**2 / 4)

    sq2 = np.sqrt(2)
    m = min(sq2, chi)

    p = -4 - m**2 / 3
    q = -2*m**3/27 - 4*m/3 + 2*m
    c1 = np.sqrt(-p / 3)
    c2 = np.arccos(1.5*q/p/c1)/3
    x1 = 2*c1*np.cos(c2 - 2*np.pi/3) + m/3

    if chi <= sq2:
        vmax = 0
    else:
        x0 = 2 *c1*np.cos(c2) + m/3
        if x0 < chi:
            vmax = (x0 - m) * _sqrt_f(x0)
        else:
            vmax = (chi - m) * _sqrt_f(chi)

    umax = _sqrt_f(m)
    vmin = (x1 - m) * x1 * np.exp(-x1**2 / 4)

    return umax, vmin, vmax


def rvs_rou_shifted_maxwell(chi, size=1, seed=None):

    def qpdf(x, chi):
        return _lazywhere((x >= 0) & (x <= chi), (x, chi),
                          lambda x, chi: x**2 * np.exp(-x**2 / 2), 0)
    umax, vmin, vmax = _get_rectangle_rou_shifted_maxwell(chi)
    m = min(np.sqrt(2), chi)

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

    return np.sqrt(1 - (y/chi)**2)
