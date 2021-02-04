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
    cdef cpinv.UNUR_DISTR * _distr
    cdef cpinv.UNUR_PAR * _par
    cdef cpinv.UNUR_GEN * _gen

    def __cinit__(self):
        # create generator for Gamma rvs using PINV
        cdef double[1] param
        param[0] = 1.5
        self._distr = cpinv.unur_distr_gamma(param, 1)
        self._par = cpinv.unur_pinv_new(self._distr)
        self._gen = cpinv.unur_init(self._par)
        if self._gen == NULL:
            raise RuntimeError('Failed to create UNURAN generator')

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
        c = cpinv.unur_distr_cont_eval_cdf(ub, self._distr)

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
            y = cpinv.unur_pinv_eval_approxinvcdf(self._gen, c*u)
            random_values[i] = math.sqrt(1.0 - y/ub)

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
        c = cpinv.unur_distr_cont_eval_cdf(ub, self._distr)

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
