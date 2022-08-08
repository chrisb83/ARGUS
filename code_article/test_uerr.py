'''
Script to test the approximation error of PINV (u- and x-error)
'''
import numpy as np
import pandas as pd
import time
from _utils import check_runtime, time_rvs
from cy_pinv import rvs_gammaincinv, argus_pinv
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# test u-error for small chi
# -----------------------------------------------------------------------------

# the u-error explodes for small values if this case is not handled separately

chi_lst = [1e-2, 1e-1, 1.0]

argus_gen = argus_pinv(uerror=1e-10)
for chi in chi_lst:
        uerr = argus_gen.check_uerror(chi, 100000)
        print('{}: {}, {}'.format(chi, np.max(uerr), np.median(uerr)))


# -----------------------------------------------------------------------------
# test approximation for small chi
# -----------------------------------------------------------------------------

N = 100000
chi_lst = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]

df_lst = []
print('Test approximation...')
for chi in chi_lst:
        res_dict = {}
        print(chi)
        # u-error
        for i in range(3):
                if i < 2:
                        uerr = argus_gen.check_uerror_approx(chi, N, newton=i)
                        mx, med = np.max(uerr), np.median(uerr)
                        descr = 'u-error (steps={})'.format(i)
                        res_dict[descr] = mx
                        print('{}: max={}, median={}'.format(descr, mx, med))
                if i > 0:
                        uerr = argus_gen.check_uerror_approx(chi, N, newton=i, exact=1)
                        mx, med = np.max(uerr), np.median(uerr)
                        descr = 'u-error (steps={}, exact)'.format(i)
                        res_dict[descr] = mx
                        print('{}: max={}, median={}'.format(descr, mx, med))

        # x-error
        for i in range(3):
                if i < 2:
                        xerr = argus_gen.check_xerror_approx(chi, N, newton=i)
                        mx, med = np.max(xerr), np.median(xerr)
                        descr = 'x-error (steps={})'.format(i)
                        res_dict[descr] = mx
                        print('{}: max={}, median={}'.format(descr, mx, med))
                if i > 0:
                        xerr = argus_gen.check_xerror_approx(chi, N, newton=i, exact=1)
                        mx, med = np.max(xerr), np.median(xerr)
                        descr = 'x-error (steps={}, exact)'.format(i)
                        res_dict[descr] = mx
                        print('{}: max={}, median={}'.format(descr, mx, med))
        print('\n')
        df_lst.append(pd.DataFrame.from_dict(res_dict, orient='index', columns=[chi]))

df = pd.concat(df_lst, axis=1)
print(df)
df.to_csv('../tables/u-x-error.csv')
df.to_latex('../tables/u-x-error.tex')
