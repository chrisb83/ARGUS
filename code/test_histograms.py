'''
Script to test that all methods generate rvs of the ARGUS distribution
by comparing the plots of the histograms to the pdf
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from rejection import rvs_rejection_xexp, rvs_rejection_beta
from rou_maxwell import  rvs_rou_maxwell, rvs_rou_shifted_maxwell
from rou_gamma import rvs_rou_gamma, rvs_rou_shifted_gamma
from cond_gamma import rvs_cond_gamma

from cy_rejection import rvs_rejection_xexp_cy, rvs_rejection_beta_cy
from cy_rou_gamma import rvs_rou_gamma_cy, rvs_rou_shifted_gamma_cy
from cy_rou_maxwell import rvs_rou_shifted_maxwell_cy, rvs_rou_maxwell_cy
from cy_pinv import (rvs_gammaincinv, argus_pinv, argus_pinv_fixed,
                     argus_pinv_fixed_direct)

chi_lst = [0.01, 0.5, 1, 1.5, 2, 2.5, 5]
rvs_pinv = argus_pinv().rvs

fcts = {rvs_rejection_xexp: 'Rejection (xexp) - Python',
        rvs_rejection_beta: 'Rejection (beta) - Python',
        rvs_rou_maxwell: 'RoU (Maxwell) - Python',
        rvs_rou_shifted_maxwell: 'RoU shifted (Maxwell) - Python',
        rvs_rou_gamma: 'RoU (Gamma) - Python',
        rvs_rou_shifted_gamma: 'RoU shifted (Gamma) - Python',
        rvs_cond_gamma: 'Conditional Gamma - Python',
        rvs_pinv: 'PINV (Gamma) - Cython',
        rvs_rejection_xexp_cy: 'Rejection (xexp) - Cython',
        rvs_rejection_beta_cy: 'Rejection (beta) - Cython',
        rvs_rou_maxwell_cy: 'RoU (Maxwell) - Cython',
        rvs_rou_shifted_maxwell_cy: 'RoU shifted (Maxwell) - Cython',
        rvs_rou_gamma_cy: 'RoU (Gamma) - Cython',
        rvs_rou_shifted_gamma_cy: 'RoU shifted (Gamma) - Cython',
        rvs_gammaincinv: 'Inversion (gammaincinv) - Cython'}

fig, ax = plt.subplots(1, 1)
for fct in fcts:
    i = 0
    for chi in chi_lst:
        # skip a few slow methods
        skip = fct.__name__ == 'rvs_rejection_beta_cy' and chi > 2
        skip = skip or (fct.__name__ == 'rvs_cond_gamma_cy' and chi < 1.5)
        skip = skip or (fct.__name__ == 'rvs_rou_maxwell_cy' and chi < 1)
        skip = skip or (fct.__name__ == 'rvs_rejection_beta' and chi > 2)
        skip = skip or (fct.__name__ == 'rvs_cond_gamma' and chi < 1.5)
        skip = skip or (fct.__name__ == 'rvs_cond_gamma_gen' and chi < 1.5)
        skip = skip or (fct.__name__ == 'rvs_rou_maxwell' and chi < 1)
        if not skip:
            r = fct(chi, size=5000)
            x = np.linspace(0, 1, 500)
            if chi <= 1.e-5:
                ax.plot(x, 3*x*np.sqrt(1-x**2), 'r-', lw=5, alpha=0.6)
            else:
                ax.plot(x, stats.argus.pdf(x, chi), 'r-', lw=5, alpha=0.6)
            ax.hist(r, bins=50, density=True, histtype='stepfilled', alpha=0.2)
            ax.set_title('{}, chi={}'.format(fcts[fct], chi))
            plt.savefig('../img/{}_chi_{}.png'.format(fct.__name__, i))
            i += 1
            ax.clear()

# test PINV with small chi

fcts = {rvs_pinv: 'PINV (Gamma) - Cython'}
chi_lst = [1e-5, 1e-4, 1e-2, 0.05, 0.5]
fig, ax = plt.subplots(1, 1)
for fct in fcts:
    i = 0
    for chi in chi_lst:
        r = fct(chi, size=5000)
        x = np.linspace(0, 1, 500)
        if chi <= 1.e-5:
            ax.plot(x, 3*x*np.sqrt(1-x**2), 'r-', lw=5, alpha=0.6)
        else:
            ax.plot(x, stats.argus.pdf(x, chi), 'r-', lw=5, alpha=0.6)
        ax.hist(r, bins=50, density=True, histtype='stepfilled', alpha=0.2)
        ax.set_title('{}, chi={}'.format(fcts[fct], chi))
        plt.savefig('../img/{}_chi_{}.png'.format(fct.__name__, i))
        i += 1
        ax.clear()

# test fixed parameter functions

chi_lst = [1e-5, 1e-4, 1e-2, 0.5, 1.0, 5.0]
fig, ax = plt.subplots(1, 1)

for m in ['gamma', 'direct']:
    i = 0
    for chi in chi_lst:
        if m == 'gamma':
            fct = argus_pinv_fixed(chi).rvs
            nm = 'argus_pinv_fixed'
        elif m == 'direct':
            fct = argus_pinv_fixed_direct(chi).rvs
            nm = 'argus_pinv_fixed_direct'

        r = fct(size=5000)
        x = np.linspace(0, 1, 500)
        if chi <= 1.e-5:
            ax.plot(x, 3*x*np.sqrt(1-x**2), 'r-', lw=5, alpha=0.6)
        else:
            ax.plot(x, stats.argus.pdf(x, chi), 'r-', lw=5, alpha=0.6)
        ax.hist(r, bins=50, density=True, histtype='stepfilled', alpha=0.2)
        ax.set_title('{}, chi={}'.format(nm, chi))
        plt.savefig('../img/{}_chi_{}.png'.format(nm, i))
        i += 1
        ax.clear()
