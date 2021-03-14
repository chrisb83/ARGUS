'''
Script to test ...
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

chi_lst = [1e-4, 1e-3, 1e-2, 1e-1]

argus_gen = argus_pinv(uerror=1e-13)
# for chi in chi_lst:
#         uerr = argus_gen.check_uerror(chi, 1000)
#         print('{}: {}, {}'.format(chi, np.max(uerr), np.median(uerr)))


# -----------------------------------------------------------------------------
# test approximation for small chi
# -----------------------------------------------------------------------------

N = 100000
#N = 3
chi_lst = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
#chi_lst = [1e-3]

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

#expr = sympy.erf(sympy.sqrt(x)) - 2*sympy.sqrt(x)*sympy.exp(-x)/sympy.sqrt(sympy.pi)
#sympy.series(expr.subs(x, y**2/2), y, 0, 9)

# chi = 1e-4
# lst = []
# argus = argus_pinv()
# for i in range(10):
#         tic = time.time()
#         x = argus.rvs(chi, size=1000000)
#         lst.append(time.time() - tic)

# print('{}: {}, {}'.format(chi, np.mean(lst), np.std(lst)))
# print((np.mean(x), np.var(x)))

# lst = []
# argus = argus_pinv()
# for i in range(10):
#         tic = time.time()
#         x = argus.rvs(chi, size=1000000, newton=1)
#         lst.append(time.time() - tic)

# print('{}: {}, {}'.format(chi, np.mean(lst), np.std(lst)))
# print((np.mean(x), np.var(x)))

# fig, ax = plt.subplots(1, 1)
# r = argus.rvs(chi, size=5000)
# x = np.linspace(0, 1, 500)
# ax.plot(x, stats.argus.pdf(x, chi), 'r-', lw=5, alpha=0.6)
# ax.hist(r, bins=50, density=True, histtype='stepfilled', alpha=0.2)
# ax.set_title('chi={}'.format(chi))
# plt.savefig('../img/approx_chi.png')
# ax.clear()