# -----------------------------------------------------------------------------
# implementation of ARGUS generators - conditional Gamma distribution
# -----------------------------------------------------------------------------

import numpy as np
from numpy.random import default_rng


def rvs_cond_gamma(chi, size=1, seed=None):
    '''
    Naive methods to sample from the Gamma distribution and condition on
    Gamma <= chi**2 / 2
    '''
    rg = default_rng(seed)
    x = np.zeros(size)
    chi2 = chi * chi
    simulated = 0
    while simulated < size:
        k = size - simulated
        g = rg.standard_gamma(1.5, size=k)
        accept = (g <= chi2 / 2)
        num_accept = np.sum(accept)
        if num_accept > 0:
            x[simulated:(simulated + num_accept)] = g[accept]
            simulated += num_accept
    return np.sqrt(1 - 2 * x / chi2)
