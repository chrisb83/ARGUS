# -----------------------------------------------------------------------------
# implementation of ARGUS generators - rejection methods
# -----------------------------------------------------------------------------

import numpy as np
from numpy.random import default_rng
from scipy import stats
from _utils import _lazywhere


# -----------------------------------------------------------------------------
# rejection from the beta density
# -----------------------------------------------------------------------------

def rvs_rejection_beta(chi, size=1, seed=None):
    x = np.zeros(size)
    d = -chi**2 / 2
    rg = default_rng(seed)
    simulated = 0
    # apply rejection method
    while simulated < size:
        k = size - simulated
        u = rg.uniform(size=k)
        v = rg.uniform(size=k)
        z = v**(2/3)
        accept = (np.log(u) <= d * z)
        num_accept = np.sum(accept)
        if num_accept > 0:
            rvs = np.sqrt(1 - z[accept])
            x[simulated:(simulated + num_accept)] = rvs
            simulated += num_accept
    return x


# -----------------------------------------------------------------------------
# rejection from the x*exp(...)
# -----------------------------------------------------------------------------

def rvs_rejection_xexp(chi, size=1, seed=None):
    x = np.zeros(size)
    chi2 = chi * chi
    echi = np.exp(-chi2 / 2)
    rg = default_rng(seed)
    simulated = 0
    while simulated < size:
        k = size - simulated
        u = rg.uniform(size=k)
        v = rg.uniform(size=k)
        z = 2 * np.log(echi * (1 - v) + v) / chi2
        accept = (u*u + z <= 0)
        num_accept = np.sum(accept)
        if num_accept > 0:
            rvs = np.sqrt(1 + z[accept])
            x[simulated:(simulated + num_accept)] = rvs
            simulated += num_accept
    return x
