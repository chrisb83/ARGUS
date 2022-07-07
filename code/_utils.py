# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

import time
import numpy as np
import pandas as pd


def time_rvs(func, args, N, rep=20, seed=None):
    if seed is not None:
        np.random.seed(seed)
    lst = []
    for i in range(rep):
        tic = time.time()
        r = func(*args, size=N)
        lst.append(time.time() - tic)
    t = np.array(lst)
    return t.mean(), t.std()


def time_rvs_varying(func, chi, N, rep=20, seed=None):
    if seed is not None:
        np.random.seed(seed)
    lst = []
    delta = 0.01*chi
    params = chi - delta + 2 * delta * np.random.uniform(size=N)
    for i in range(rep):
        tic = time.time()
        r = func(params, size=N)
        lst.append(time.time() - tic)
    t = np.array(lst)
    return t.mean(), t.std()


def check_runtime(fcts, chi_lst, cfg, print_output=False, varying=False):
    df_lst = []
    for N in cfg.keys():
        res = {}
        niter = cfg[N]
        if print_output:
            print('Starting test with N={}'.format(N))
        for fct, fct_name in fcts.items():
            if print_output:
                print(fct_name)
            res_chi = {}
            for chi in chi_lst:
                # skip some of the very slow items
                skip = fct.__name__ == 'rvs_rejection_beta_cy' and chi > 2
                skip = skip or (fct.__name__ == 'rvs_cond_gamma_cy' and chi < 1.5)
                skip = skip or (fct.__name__ == 'rvs_rou_maxwell_cy' and chi < 1)
                skip = skip or (fct.__name__ == 'rvs_rejection_beta' and chi > 2)
                skip = skip or (fct.__name__ == 'rvs_cond_gamma' and chi < 1.5)
                skip = skip or (fct.__name__ == 'rvs_gammaincinv' and chi != 1.0)
                skip = skip or (fct.__name__ == 'rvs_cond_gamma_gen' and chi < 1.5)
                skip = skip or (fct.__name__ == 'rvs_rou_maxwell' and chi < 1)
                if skip:
                    m, s = -1, 0
                else:
                    if varying:
                        m, s = time_rvs_varying(fct, chi, N, rep=niter, seed=12345)
                    else:
                        m, s = time_rvs(fct, (chi,), N, rep=niter, seed=12345)
                res_chi[chi] = (m*1000, s*1000)
                if print_output:
                    print("chi = {}: {} +/- {}".format(chi, round(m, 3), round(s, 3)))
            res[fct_name] = res_chi

        res2 = {}
        for k, v in res.items():
            res_temp = {}
            for chi, v2 in v.items():
                m, s = int(round(v2[0])), int(round(v2[1]))
                res_temp[round(chi, 4)] = '{} +/- {}'.format(m, s)
            res2[k] = res_temp
        df_temp = pd.DataFrame.from_dict(res2, orient='index')
        df_temp['N'] = N
        df_temp['iter'] = niter
        df_lst.append(df_temp)

    return pd.concat(df_lst)


def check_runtime_fixed(cls_dict, chi_lst, cfg, print_output=False):
    df_lst = []
    for N in cfg.keys():
        res = {}
        niter = cfg[N]
        if print_output:
            print('Starting test with N={}'.format(N))
        for cls, cls_name in cls_dict.items():
            if print_output:
                print(cls_name)
            res_chi = {}
            for chi in chi_lst:
                # set up rvs method depending on chi. fct does not take chi
                # as an argument, just size
                fct = cls(chi).rvs
                m, s = time_rvs(fct, (), N, rep=niter, seed=12345)
                res_chi[chi] = (m*1000, s*1000)
                if print_output:
                    print("chi = {}: {} +/- {}".format(chi, round(m, 3), round(s, 3)))
            res[cls_name] = res_chi

        res2 = {}
        for k, v in res.items():
            res_temp = {}
            for chi, v2 in v.items():
                m, s = int(round(v2[0])), int(round(v2[1]))
                res_temp[round(chi, 4)] = '{} +/- {}'.format(m, s)
            res2[k] = res_temp
        df_temp = pd.DataFrame.from_dict(res2, orient='index')
        df_temp['N'] = N
        df_temp['iter'] = niter
        df_lst.append(df_temp)

    return pd.concat(df_lst)


# taken from from scipy._lib._util
def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).
    Examples
    --------
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])
    Notice, it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.
    """
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    cond = np.asarray(cond)
    arrays = np.broadcast_arrays(*arrays)
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out
