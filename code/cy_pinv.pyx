# -----------------------------------------------------------------------------
# implementation of ARGUS generators - numerical inversion
# -----------------------------------------------------------------------------
import numpy as np
cimport numpy as np
cimport cpinv
cimport cython
from libc cimport math
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from libc.stdint cimport uint16_t, uint64_t
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from scipy.special cimport cython_special


# -----------------------------------------------------------------------------
# PINV (UNU.RAN)
# -----------------------------------------------------------------------------

cdef class argus_pinv:
    cdef cpinv.UNUR_DISTR * _distr0
    cdef cpinv.UNUR_DISTR * _distr1
    cdef cpinv.UNUR_DISTR * _distr2
    cdef cpinv.UNUR_PAR * _par0
    cdef cpinv.UNUR_PAR * _par1
    cdef cpinv.UNUR_PAR * _par2
    cdef cpinv.UNUR_GEN * _gen0
    cdef cpinv.UNUR_GEN * _gen1
    cdef cpinv.UNUR_GEN * _gen2
    cdef double c1, c2

    def __cinit__(self, double uerror=1.e-10):
        cdef double[1] param
        param[0] = 1.5

        # PINV to be used for [1, infty]
        self._distr0 = cpinv.unur_distr_gamma(param, 1)
        self._par0 = cpinv.unur_pinv_new(self._distr0)
        cpinv.unur_pinv_set_u_resolution(self._par0, uerror)
        self._gen0 = cpinv.unur_init(self._par0)
        if self._gen0 == NULL:
            raise RuntimeError('Failed to create UNURAN generator')

        # PINV to be used for [0.1, 1] => condition Gamma on 0.5
        self._distr1 = cpinv.unur_distr_gamma(param, 1)
        cpinv.unur_distr_cont_set_domain(self._distr1, 0.0, 0.5)
        self._par1 = cpinv.unur_pinv_new(self._distr1)
        cpinv.unur_pinv_set_u_resolution(self._par1, uerror*1e-3)
        self._gen1 = cpinv.unur_init(self._par1)
        if self._gen1 == NULL:
            raise RuntimeError('Failed to create UNURAN generator')

        # PINV to be used for [0.01, 0.1] => conditio Gamma on 0.1**2/2
        self._distr2 = cpinv.unur_distr_gamma(param, 1)
        cpinv.unur_distr_cont_set_domain(self._distr2, 0.0, 0.005)
        self._par2 = cpinv.unur_pinv_new(self._distr2)
        cpinv.unur_pinv_set_u_resolution(self._par2, uerror*1e-3)
        self._gen2 = cpinv.unur_init(self._par2)
        if self._gen2 == NULL:
            raise RuntimeError('Failed to create UNURAN generator')

        # constants P[Gamma(1.5) <= chi**2/2] for chi in [0.1, 0.01]
        self.c1 = 0.19874804309879915
        self.c2 = 0.0002651650586556101

    def __dealloc__(self):
        if self._gen0 is not NULL:
            cpinv.unur_free(self._gen0)
        if self._gen1 is not NULL:
            cpinv.unur_free(self._gen1)
        if self._gen2 is not NULL:
            cpinv.unur_free(self._gen2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def rvs(self, double chi, Py_ssize_t size=1):
        cdef double u, ub, c, y, v = 0
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values
        cdef int steps

        if chi <= 0:
            raise ValueError('chi must be > 0')

        # compute P[gamma(1.5) <= chi**2/2]
        ub = chi*chi / 2.0
        c = cython_special.gammainc(1.5, ub)

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(size, dtype='float64')

        if chi > 1.e-5 and chi <= 0.01:
            v = math.sqrt(2.0*math.pi)*c/(2.0*ub*chi)
        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(size):
            u = rng.next_double(rng.state)
            if chi <= 0.01:
                y = ub * math.pow(u, 2./3.)
                # y = ub * math.exp((2./3.)*math.log(u))
                if chi > 1.e-5:
                    # do a single Newton iteration
                    ey = 1.0 + y*(1.0 + y*(0.5 + y/6.0))
                    y = y*(ey*(1.0/3.0 - 0.1*y + v) - y*(0.5 + y/6.0))
            elif chi <= 0.1:
                y = cpinv.unur_pinv_eval_approxinvcdf(self._gen2, c*u/self.c2)
            elif chi <= 1.0:
                y = cpinv.unur_pinv_eval_approxinvcdf(self._gen1, c*u/self.c1)
            else:
                y = cpinv.unur_pinv_eval_approxinvcdf(self._gen0, c*u)
            random_values[i] = math.sqrt(1.0 - y/ub)

        return np.asarray(random_values)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def check_uerror(self, double chi, Py_ssize_t nsamples):
        cdef double u, ub, c, y
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] uerr

        if chi <= 0:
            raise ValueError('chi must be > 0')

        # compute P[gamma(1.5) <= chi**2/2]
        ub = chi*chi / 2.0
        c = cython_special.gammainc(1.5, ub)

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        uerr = np.empty(nsamples, dtype='float64')

        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(nsamples):
            u = rng.next_double(rng.state)
            y = cpinv.unur_pinv_eval_approxinvcdf(self._gen0, c*u)
            uerr[i] = math.fabs(u - cython_special.gammainc(1.5, y) / c)
        return np.array(uerr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def check_uerror_approx(self, double chi, Py_ssize_t nsamples, int newton=0, int exact=0):
        # for small chi, approximate pdf of Gamma(1.5) conditioned on
        # [0, chi^2/2] by const * sqrt(u)

        cdef double u, ub, c, y, ey, v
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] uerr
        cdef int steps = 0

        if chi <= 0:
            raise ValueError('chi must be > 0')

        # compute P[gamma(1.5) <= chi**2/2]
        ub = chi*chi / 2.0
        c = cython_special.gammainc(1.5, ub)

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        uerr = np.empty(nsamples, dtype='float64')

        if exact != 1:
            v = math.sqrt(2.0*math.pi)*c/(2.0*ub*chi)
        for i in range(nsamples):
            u = rng.next_double(rng.state)
            y = ub * u**(2./3.)
            if newton > 0:
                steps = newton
                while steps > 0:
                    if exact == 1:
                        ey = (cython_special.gammainc(1.5, y) / c - u)
                        y = y - math.sqrt(math.pi) * c * ey / (2 * math.sqrt(y) * math.exp(-y))
                    else:
                        ey = 1.0 + y*(1.0 + y*(0.5 + y/6.0)) # math.exp(y)
                        #y = y*(1 + ey) - cython_special.expm1(y) - y*ey*(y/2.0 + 2.0*ub)/5.0
                        #y = y*(ey - y/2.0) - y*ey*(y/2.0 + 2.0*ub)/5.0
                        # y = y*(ey*(1 - 0.1*y -0.4*ub) - y/2.0) # THIS ONE
                        y = y*(ey*(1.0/3.0 - 0.1*y + v) - y*(0.5 + y/6.0))
                    steps -= 1
            uerr[i] = math.fabs(u - cython_special.gammainc(1.5, y) / c)
        return np.array(uerr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def check_xerror_approx(self, double chi, Py_ssize_t nsamples, int newton=0, int exact=0):
        # for small chi, approximate pdf of Gamma(1.5) conditioned on
        # [0, chi^2/2] by const * sqrt(u)

        cdef double u, ub, c, x1, x2, ey, v
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] uerr
        cdef int steps

        if chi <= 0:
            raise ValueError('chi must be > 0')

        # compute P[gamma(1.5) <= chi**2/2]
        ub = chi*chi / 2.0
        c = cython_special.gammainc(1.5, ub)

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        uerr = np.empty(nsamples, dtype='float64')

        if exact != 1:
            v = math.sqrt(2.0*math.pi)*c/(2.0*ub*chi)

        for i in range(nsamples):
            u = rng.next_double(rng.state)
            x1 = ub * u**(2./3.)
            if newton > 0:
                steps = newton
                while steps > 0:
                    if exact == 1:
                        ey = (cython_special.gammainc(1.5, x1) / c - u)
                        x1 = x1 - math.sqrt(math.pi) * c * ey / (2 * math.sqrt(x1) * math.exp(-x1))
                    else:
                        ey = 1.0 + x1*(1.0 + x1*(0.5 + x1/6.0)) # math.exp(x1)
                        x1 = x1*(ey*(1.0/3.0 - 0.1*x1 + v) - x1*(0.5 + x1/6.0))
                    steps -= 1
            x2 = cython_special.gammaincinv(1.5, c*u)
            uerr[i] = math.fabs(x1 - x2)
        return np.array(uerr)


# -----------------------------------------------------------------------------
# PINV (UNU.RAN) for fixed parameter case
# -----------------------------------------------------------------------------

cdef class argus_pinv_fixed:
    cdef cpinv.UNUR_DISTR * _distr
    cdef cpinv.UNUR_PAR * _par
    cdef cpinv.UNUR_GEN * _gen
    cdef double ub

    def __cinit__(self, double chi, double uerror=1.e-10):
        cdef double[1] param
        param[0] = 1.5

        if chi <= 0:
            raise ValueError('chi must be > 0')

        self.ub = chi*chi/2.0
        # generator for Gamma rvs conditional on [0, chi^2 / 2]
        self._distr = cpinv.unur_distr_gamma(param, 1)
        cpinv.unur_distr_cont_set_domain(self._distr, 0.0, self.ub)
        self._par = cpinv.unur_pinv_new(self._distr)
        cpinv.unur_pinv_set_u_resolution(self._par, uerror)
        self._gen = cpinv.unur_init(self._par)
        if self._gen == NULL:
            raise RuntimeError('Failed to create UNURAN generator')


    def __dealloc__(self):
        if self._gen is not NULL:
            cpinv.unur_free(self._gen)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def rvs(self, Py_ssize_t size=1):
        cdef double u, y
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values
        cdef int steps

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(size, dtype='float64')

        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(size):
            u = rng.next_double(rng.state)
            y = cpinv.unur_pinv_eval_approxinvcdf(self._gen, u)
            random_values[i] = math.sqrt(1.0 - y/self.ub)

        return np.asarray(random_values)



# -----------------------------------------------------------------------------
# PINV (UNU.RAN) for fixed parameter case using the ARGUS density
# -----------------------------------------------------------------------------

cdef double argus_quasipdf(double x, const cpinv.UNUR_DISTR *distr):
    # return value proportional to the ARGUS PDF
    # normalizing constants are ignored
    cdef double y, chi
    cdef const double *fpar
    cdef int npar
    if x >= 1.0 or x <= 0.0:
        return 0.0
    npar = cpinv.unur_distr_cont_get_pdfparams(distr, &fpar)
    chi = fpar[0]
    y = 1 - x*x
    return x * math.sqrt(y) * math.exp(-0.5*chi*chi*y)

cdef class argus_pinv_fixed_direct:
    cdef cpinv.UNUR_DISTR * _distr
    cdef cpinv.UNUR_PAR * _par
    cdef cpinv.UNUR_GEN * _gen
    cdef double ub

    def __cinit__(self, double chi, double uerror=1.e-10):
        cdef double[1] param
        param[0] = chi

        if chi <= 0:
            raise ValueError('chi must be > 0')

        self._distr = cpinv.unur_distr_cont_new()
        cpinv.unur_distr_cont_set_pdf(self._distr, argus_quasipdf)
        cpinv.unur_distr_cont_set_pdfparams(self._distr, param, 1)
        cpinv.unur_distr_cont_set_domain(self._distr, 0.0, 1.0)
        self._par = cpinv.unur_pinv_new(self._distr)
        cpinv.unur_pinv_set_u_resolution(self._par, uerror)
        self._gen = cpinv.unur_init(self._par)
        if self._gen == NULL:
            raise RuntimeError('Failed to create UNURAN generator')


    def __dealloc__(self):
        if self._gen is not NULL:
            cpinv.unur_free(self._gen)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def rvs(self, Py_ssize_t size=1):
        cdef double u
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values
        cdef int steps

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(size, dtype='float64')

        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(size):
            u = rng.next_double(rng.state)
            random_values[i] = cpinv.unur_pinv_eval_approxinvcdf(self._gen, u)

        return np.asarray(random_values)

# -----------------------------------------------------------------------------
# HINV (UNU.RAN)
# -----------------------------------------------------------------------------

cdef class argus_hinv:
    cdef cpinv.UNUR_DISTR * _distr
    cdef cpinv.UNUR_PAR * _par
    cdef cpinv.UNUR_GEN * _gen

    def __cinit__(self):
        # create generator for Gamma rvs using PINV
        cdef double[1] param
        param[0] = 1.5
        self._distr = cpinv.unur_distr_gamma(param, 1)
        self._par = cpinv.unur_hinv_new(self._distr)
        self._gen = cpinv.unur_init(self._par)
        if self._gen == NULL:
            raise RuntimeError('Failed to create UNURAN generator')

    def __dealloc__(self):
        if self._gen is not NULL:
            cpinv.unur_free(self._gen)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def rvs(self, double chi, Py_ssize_t size=1):
        cdef double u, ub, c, y
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values

        if chi <= 0:
            raise ValueError('chi must be > 0')

        # compute P[gamma(1.5) <= chi**2/2]
        ub = chi*chi / 2.0
        c = cython_special.gammainc(1.5, ub)

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(size, dtype='float64')

        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(size):
            u = rng.next_double(rng.state)
            y = cpinv.unur_hinv_eval_approxinvcdf(self._gen, c*u)
            random_values[i] = math.sqrt(1.0 - y/ub)

        return np.asarray(random_values)


# -----------------------------------------------------------------------------
# (slow) approach using inverse of Gamma CDF gammaincinv
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_gammaincinv(double chi, Py_ssize_t size=1):
    cdef double u, ub, c, y
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double[::1] random_values

    # compute P[gamma(1.5) <= chi**2/2]
    ub = chi*chi / 2.0
    c = cython_special.gammainc(1.5, ub)

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(size, dtype='float64')

    # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
    for i in range(size):
        u = rng.next_double(rng.state)
        y = cython_special.gammaincinv(1.5, c*u)
        random_values[i] = math.sqrt(1.0 - y/ub)

    return np.asarray(random_values)
