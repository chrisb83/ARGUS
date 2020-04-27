'''
Script to test the runtime of the different algorithms. For the chosen
parameter values, repeatedly generate the specified number of samples
and measure the required time. The average and standard deviation of the
runtime (in milliseconds, mean and standard deviation) are saved in table
for each algorithm (csv and tex)
'''
import numpy as np
import pandas as pd
import time
from _utils import check_runtime, time_rvs
from rejection import rvs_rejection_xexp, rvs_rejection_beta
from rou_maxwell import  rvs_rou_maxwell, rvs_rou_shifted_maxwell
from rou_gamma import rvs_rou_gamma, rvs_rou_shifted_gamma
from cond_gamma import rvs_cond_gamma

from cy_rejection import rvs_rejection_xexp_cy, rvs_rejection_beta_cy
from cy_rou_gamma import rvs_rou_gamma_cy, rvs_rou_shifted_gamma_cy
from cy_rou_maxwell import rvs_rou_shifted_maxwell_cy, rvs_rou_maxwell_cy
from cy_pinv import rvs_gammaincinv, argus_pinv

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------

# parameters to test
chi_lst = [0.01, 0.5, 1, 1.5, 2, 2.5, 5]
# dictionary, keys: sample size, values: number of repetitions
cfg = {100000: 100,
       1000000: 100}

# -----------------------------------------------------------------------------
# test main parameters (Python)
# -----------------------------------------------------------------------------

print('----------------------------------------')
print('Test runtime of Python implementation...')
print('----------------------------------------\n')

fcts = {rvs_rejection_xexp: 'Rejection (xexp)',
        rvs_rejection_beta: 'Rejection (beta)',
        rvs_rou_maxwell: 'RoU (Maxwell)',
        rvs_rou_shifted_maxwell: 'RoU shifted (Maxwell)',
        rvs_rou_gamma: 'RoU (Gamma)',
        rvs_rou_shifted_gamma: 'RoU shifted (Gamma)',
        rvs_cond_gamma: 'Conditional Gamma'}

df_py = check_runtime(fcts, chi_lst, cfg, print_output=True)
df_py['Language'] = 'Python'

# -----------------------------------------------------------------------------
# test main parameters (Cython)
# -----------------------------------------------------------------------------
print('\n\n----------------------------------------')
print('Test runtime of Cython implementation...')
print('----------------------------------------\n')
# test setup time of PINV
lst = []
for i in range(100):
        tic = time.time()
        argus = argus_pinv()
        lst.append(time.time() - tic)

m, v = np.mean(lst), np.std(lst)
print('Setup time for PINV: {} +/- {}'.format(1000*m, 1000*v))

# use rvs after the setup to measure generation times w/o setup
rvs_pinv = argus.rvs

fcts = {rvs_pinv: 'PINV (Gamma)',
        rvs_rejection_xexp_cy: 'Rejection (xexp)',
        rvs_rejection_beta_cy: 'Rejection (beta)',
        rvs_rou_maxwell_cy: 'RoU (Maxwell)',
        rvs_rou_shifted_maxwell_cy: 'RoU shifted (Maxwell)',
        rvs_rou_gamma_cy: 'RoU (Gamma)',
        rvs_rou_shifted_gamma_cy: 'RoU shifted (Gamma)',
        rvs_gammaincinv: 'Inversion (gammaincinv)'}

df_cy = check_runtime(fcts, chi_lst, cfg, print_output=True)
df_cy['Language'] = 'Cython'

# -----------------------------------------------------------------------------
# save results
# -----------------------------------------------------------------------------

df = pd.concat([df_py, df_cy])
df.to_csv("../tables/argus_generation_times.csv")
df.to_latex("../tables/argus_generation_times.tex")

# -----------------------------------------------------------------------------
# take beta, gamma and chisquare in NumpPy as a benchmark
# -----------------------------------------------------------------------------

print('\n\n----------------------------------------')
print('Benchmark against NumPy Generator ...')
print('----------------------------------------\n')
distr = {'beta': (1.5, 1),
         'chisquare': (3, ),
         'standard_gamma': (1.5, )}

rg = np.random.default_rng()
res = {}
for d, args in distr.items():
        print('Starting test with {}'.format(d))
        res_tmp = {}
        for size, niter in cfg.items():
                # the distribution is an attribute of the generator in Numpy
                fct = getattr(rg, d)
                m, s = time_rvs(fct, args, N=size, rep=niter)
                res_tmp[size] = np.round((m*1000, s*1000), 3)
                print("size={}: {} +/- {}".format(size, round(m, 3), round(s, 3)))
        res[d] = res_tmp

df = pd.DataFrame.from_dict(res, orient='index')
df.to_csv("../tables/numpy_benchmark.csv")
