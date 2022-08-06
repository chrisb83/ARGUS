# This is a pure Python implementation of Algorithm 1 and 2 in the article that
# requires SciPy >= 1.8.0
# examples of how to use the algorithms are provided at the end of the file
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
    """ This is the main algorithm (Algorithm 2) that can be used to sample
    for all values of chi. To sample random variates for a given chi, pass a
    scalar chi to the method rvs. The setup is only run once when the object is
    instantiated. Calling rvs with differnt chis does not require to run the
    setup again.
    If one needs to sample random variates for different values of chi in one
    go, use ArgusVarPar
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
        """ chi needs to be a scalar.
        """

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


class ArgusVarPar:

    """ This is the main algorithm (Algorithm 2 in the article) that can be
    used to sample for all values of chi.
	Compared to ArgusGeneral, it allows to sample random variates for different
    values of chi in one go, see the example in the method rvs.
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
            u_resolution=uerror*0.1
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

    def _rgam15chiGr1(self, chi, rng):
        # for chi values > 1
        CU = gammainc(1.5, chi*chi*0.5) * (1-rng.uniform(size=chi.shape))
        return self._gen0.ppf(CU)

    def _rgam15chiGr01(self, chi, rng):
        # for chi values > .1
        CU = gammainc(1.5, chi*chi*0.5)/self.c1 * (1-rng.uniform(size=chi.shape))
        return self._gen1.ppf(CU)

    def _rgam15chiGr001(self, chi, rng):
        # for chi values > .01
        CU = gammainc(1.5, chi*chi*0.5)/self.c2 * (1-rng.uniform(size=chi.shape))
        return self._gen2.ppf(CU)

    def _rgam15chiLe001(self, chi, rng):
        # for chi values < .01
        y = chi*chi*0.5 * (1-rng.uniform(size=chi.shape))**(2/3)
        iv = chi > 1.e-5
        Y = y[iv]
        # use approximation (16) of the paper
        term1 = (Y*(1+Y*(1+Y*(0.5+Y/6))))*(1/3-0.1*Y+2.*(1/3-0.1*chi[iv]**2))
        Y = term1 - 0.5*Y*Y*(1+Y/3)
        y[iv] = Y
        return y

    def rvs(self, chiv, seed=None):
        """
        Generates one ARGUS rv for each entry of chiv
        chiv: array_like

        Example:
        Create 1000 rvs with chi=0.1, 1.3, 3.5

        >>> chi_arr = np.array([0.1, 1.3, 3.5]) * np.ones((1000, 3))
        >>> gen = ArgusInvVarPar()
        >>> r = gen.rvs(chi_arr)

        Each column in r contains 1000 samples for one of the values of chi.
        Take mean over columns:

        >>> r.mean(axis=0)

        The sample should be close to the actual mean that can be computed as
        follows:

        >>> from scipy import stats
        >>> stats.argus([0.1, 1.3, 3.5]).mean()
        """

        if seed is None:
            rng = self._urng
        else:
            rng = np.random.default_rng(seed)

        chiv = np.array(chiv)
        y = np.empty_like(chiv)

        iv = chiv > 1
        y[iv]= self._rgam15chiGr1(chiv[iv], rng)

        iv = np.logical_and(chiv <= 1, chiv > 0.1)
        y[iv]= self._rgam15chiGr01(chiv[iv], rng)

        iv = np.logical_and(chiv <= .1, chiv > 0.01)
        y[iv]= self._rgam15chiGr001(chiv[iv], rng)

        iv = chiv <= .01
        y[iv]= self._rgam15chiLe001(chiv[iv], rng)

        return np.sqrt(1.0 - 2*y/(chiv*chiv))


class ArgusGamma:
    """ This is the algorithm that can be used for fixed values of chi.
    It is more robust than ArgusDirect for large values of chi.
    It is the part of Algorithm 1 in the paper that is used for chi > 5.
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
    It is the part of Algorithm 1 in the article that applies PINV directly to
    the ARGUS density for chi <= 5. It is faster than ArgusGamma
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

###############################################################################
### some examples of how to use the algorithms ################################
###############################################################################

# create some histograms for different values of chi

import matplotlib.pyplot as plt
import time

x = np.linspace(1e-5, 0.999, num=100)
gen = ArgusGeneral()
for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    r = gen.rvs(chi, size=10000)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

gen = ArgusVarPar()
for chi in [1e-6, 0.005, 0.05, 0.5, 1, 3]:
    chiv = chi*np.ones(10000)
    y = gen.rvs(chiv=chiv)
    plt.hist(y, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    gen = ArgusDirect(chi)
    r = gen.rvs(size=10000)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    gen = ArgusGamma(chi)
    r = gen.rvs(size=10000)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

### check the u-error of Algorithm 2
gen = ArgusVarPar(uerror=1e-10)
rng = np.random.default_rng()

def cdf_argus(x=0.5,chi=1):
    return 1.0 - gammainc(1.5, 0.5*chi*chi*(1-x*x))/gammainc(1.5, 0.5*chi*chi)

def check_uerror_time(n, chi_min=0., chi_max=6.):
    """
    helper function to compute the u-error of Algorithm 2 over a range of chi
    and to show the speed of the varying parameter case
    """
    # create random parameters over the specified range
    chiv = chi_min + (chi_max - chi_min) * rng.uniform(size=n)
    # check sampling time
    start_time = time.time()
    y = gen.rvs(chiv=chiv, seed=123)
    print(f"For chi in ({chi_min}, {chi_max}): ")
    print("time={} for n={}".format(time.time() - start_time, np.size(y)))
    # compute the u-error
    Fy = cdf_argus(x=y, chi=chiv)
    urng = np.random.default_rng(seed=123)
    u = urng.uniform(size=n)
    print("the maximal observed u-error =", max(np.abs(u-Fy)))

N = 1_000_000
check_uerror_time(n=N, chi_min=1., chi_max=6.)
check_uerror_time(n=N, chi_min=0.1, chi_max=1.)
check_uerror_time(n=N, chi_min=0.01, chi_max=0.1)
check_uerror_time(n=N, chi_min=0., chi_max=0.01)

# give an example of how to sample for different chi in one go using ArgusVarPar
chis = [0.1, 1.3, 3.5]
chi_arr = np.array(chis) * np.ones((1000, 3))
y = gen.rvs(chi_arr)

# first column contains the rvs for chi=0.1, second for 1.3, third for 3.5
for i, chi in enumerate(chis):
    plt.hist(y[:, i], bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()
