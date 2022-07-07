# requires SciPy >= 1.8.0
import numpy as np
import math
from scipy import stats
from scipy.stats.sampling import NumericalInversePolynomial
from scipy.special import gammainc


class GammaDist:
    """ Define simple class with pdf/support of the Gamma distribution.
    """
    def __init__(self, p):
        self.p = p

    def pdf(self, x):
        # a normalizing constant is not needed
        return x**(self.p - 1) * math.exp(-x)

    def support(self):
        return 0, np.infty

class ArgusDist:
    """ Define simple class with pdf/support of the ARGUS distribution.
    """
    def __init__(self, chi):
        self.chi = chi
        self.chi2 = chi**2

    def pdf(self, x):
        # a normalizing constant is not needed
        y = 1 - x*x
        return x * math.sqrt(y) * math.exp(-0.5*self.chi2*y)

    def support(self):
        return 0, 1


class ArgusGeneral:
    """ This is the main algorithm that can be used to sample for all
    values of chi.
    """
    def __init__(self, uerror=1e-10, seed=None):
        # set uniform random number generator. there is no need to pass it to
        # NumericalInversePolynomial since we never use its rvs method
        # it just allows to fix the seed of the generator used in self.rvs
        self._urng = np.random.default_rng(seed)

        # PINV to be used for [1, infty]
        dist = GammaDist(1.5)
        self._gen0 = NumericalInversePolynomial(
            dist,
            center=0.5,
            domain=(0, np.infty),
            u_resolution=uerror
        )

        # PINV to be used for [0.1, 1] => condition Gamma on 0.5
        self._gen1 = NumericalInversePolynomial(
            dist,
            domain=(0, 0.5),
            u_resolution=uerror*1e-3
        )

        # PINV to be used for [0.01, 0.1] => condition Gamma on 0.1**2/2
        self._gen2 = NumericalInversePolynomial(
            dist,
            domain=(0, 0.005),
            u_resolution=uerror*1e-3
        )

        # constants P[Gamma(1.5) <= chi**2/2] for chi = 0.1, 0.01
        self.c1 = 0.19874804309879915
        self.c2 = 0.0002651650586556101

    def rvs(self, chi, size=1, seed=None):

        if chi <= 0:
            raise ValueError('chi must be > 0')

        # compute P[gamma(1.5) <= chi**2/2]
        ub = chi*chi / 2.0
        c = gammainc(1.5, ub)

        if seed is None:
            rng = self._urng
        else:
            rng = np.random.default_rng(seed)

        if chi > 1.e-5 and chi <= 0.01:
            v = np.sqrt(2.0*np.pi)*c/(2.0*ub*chi)
        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        u = rng.uniform(size=size)
        if chi <= 0.01:
            y = ub * u**(2/3)
            if chi > 1.e-5:
                # do a single Newton iteration
                ey = 1.0 + y*(1.0 + y*(0.5 + y/6.0))
                y = y*(ey*(1.0/3.0 - 0.1*y + v) - y*(0.5 + y/6.0))
        elif chi <= 0.1:
            y = self._gen2.ppf(c*u/self.c2)
        elif chi <= 1.0:
            y = self._gen1.ppf(c*u/self.c1)
        else:
            y = self._gen0.ppf(c*u)

        return np.sqrt(1.0 - y/ub)


class ArgusGamma:
    """ This is the algorithm that can be used for fixed values of chi.
    It is more robust than ArgusDirect for large values of chi.
    It is recommended to use it for chi > 5.
    """
    def __init__(self, chi, uerror=1e-10, seed=None):
        # set uniform random number generator. there is no need to pass it to
        # NumericalInversePolynomial since we never use its rvs method
        # it just allows to fix the seed of the generator used in self.rvs
        self._urng = np.random.default_rng(seed)

        if chi <= 0:
            raise ValueError('chi must be > 0')
        self.chi = chi
        self.ub = chi*chi/2.0

        dist = GammaDist(1.5)
        self._gen = NumericalInversePolynomial(
            dist,
            domain=(0, self.ub),
            u_resolution=uerror
        )

    def rvs(self, size=1, seed=None):
        if seed is None:
            rng = self._urng
        else:
            rng = np.random.default_rng(seed)
        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        u = rng.uniform(size=size)
        y = self._gen.ppf(u)
        return np.sqrt(1.0 - y/self.ub)


class ArgusDirect:
    """ This is the algorithm that can be used for fixed values of chi.
    It applies PINV directly to the ARGUS density. It is faster than ArgusGamma
    for small values of chi. For numerical accuracy, it is recommended to use
    ArgusGamma for chi > 5 instead of ArgusDirect.
    """
    def __init__(self, chi, uerror=1e-10, seed=None):
        # set uniform random number generator. there is no need to pass it to
        # NumericalInversePolynomial since we never use its rvs method
        # it just allows to fix the seed of the generator used in self.rvs
        self._urng = np.random.default_rng(seed)

        if chi <= 0:
            raise ValueError('chi must be > 0')
        self.chi = chi

        dist = ArgusDist(self.chi)
        self._gen = NumericalInversePolynomial(
            dist,
            domain=(0, 1),
            u_resolution=uerror
        )

    def rvs(self, size=1, seed=None):
        if seed is None:
            rng = self._urng
        else:
            rng = np.random.default_rng(seed)
        u = rng.uniform(size=size)
        return self._gen.ppf(u)


import matplotlib.pyplot as plt

gen1 = ArgusGeneral()
for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    r = gen1.rvs(chi, size=10000)
    x = np.linspace(1e-5, 0.999, num=100)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()


for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    gen = ArgusDirect(chi)
    r = gen.rvs(size=10000)
    x = np.linspace(1e-5, 0.999, num=100)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    gen = ArgusGamma(chi)
    r = gen.rvs(size=10000)
    x = np.linspace(1e-5, 0.999, num=100)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

%timeit ArgusGen()
%timeit gen1.rvs(0.7, size=1000000)
%timeit gen1.rvs(1e-6, size=1000000)
%timeit gen1.rvs(3.5, size=1000000)