# -----------------------------------------------------------------------------
# adapt the code from cy_pinv.pyx and cy_rou_gamma.pyx such that the
# varying parameter case can be handled
# TODO: integrate methods into the classes in cy_pinv.pyx / cy_rou_gamma.pyx
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

        # PINV to be used for [0.01, 0.1] => condition Gamma on 0.1**2/2
        self._distr2 = cpinv.unur_distr_gamma(param, 1)
        cpinv.unur_distr_cont_set_domain(self._distr2, 0.0, 0.005)
        self._par2 = cpinv.unur_pinv_new(self._distr2)
        cpinv.unur_pinv_set_u_resolution(self._par2, uerror*1e-3)
        self._gen2 = cpinv.unur_init(self._par2)
        if self._gen2 == NULL:
            raise RuntimeError('Failed to create UNURAN generator')

        # constants P[Gamma(1.5) <= chi**2/2] for chi = 0.1, 0.01
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
    def test_varying(self, double[:] params, Py_ssize_t size):
        cdef double u, ub, c, y, chi, v, sqrt2pi
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(size, dtype='float64')

        v = 0.0
        sqrt2pi = math.sqrt(2.0/math.M_PI)
        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(size):
            chi = params[i]
            ub = chi*chi / 2.0
            c = cython_special.erf(chi/math.sqrt(2)) - chi*math.exp(-ub)*sqrt2pi
            if chi > 1.e-5 and chi <= 0.01:
                v = math.sqrt(2.0*math.pi)*c/(2.0*ub*chi)
            u = rng.next_double(rng.state)
            if chi <= 0.01:
                y = ub * math.pow(u, 2./3.)
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


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms with mode shift
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def test_rou_shifted_gamma_varying(double[:] params, Py_ssize_t size):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmin, vmax, umax, z, chi2, m, x_min, x_max, chi
    cdef double[::1] random_values

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(size, dtype='float64')

    with x.lock, nogil:
        for i in range(size):
            chi = params[i]
            chi2 = chi * chi
            if chi <= 1:
                m = chi2 / 2
            else:
                m = 0.5

            if chi >= 1:
                x_max = 1.5 + math.sqrt(2)
                if chi2 / 2 < x_max:
                    x_max = chi2 / 2
                x_min = 1.5 - math.sqrt(2)
                vmax = (x_max - m) * x_max**0.25 * math.exp(-0.5 * x_max)
            else:
                vmax = 0
                x_min = m/2 - math.sqrt(4*m*m + 12*m + 25)/4. + 1.25

            umax = m**0.25 * math.exp(-0.5 * m)
            vmin = (x_min - m) * x_min**0.25 * math.exp(-0.5 * x_min)
            while (1):
                u = rng.next_double(rng.state) * umax
                v = vmin + rng.next_double(rng.state) * (vmax - vmin)
                z = v / u + m
                if 0 < z < chi2 / 2:
                    if u * u <= math.sqrt(z) * math.exp(-z):
                        break
            random_values[i] = math.sqrt(1 - 2 * z / chi2)

    return np.asarray(random_values)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def test_rou_shifted_gamma_varying_max(double chi_max, Py_ssize_t size):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmin, vmax, umax, z, chi2, m, x_min, x_max, chi
    cdef double[::1] random_values

    chi = chi_max
    chi2 = chi * chi
    if chi <= 1:
        m = chi2 / 2
    else:
        m = 0.5

    if chi >= 1:
        x_max = 1.5 + math.sqrt(2)
        if chi2 / 2 < x_max:
            x_max = chi2 / 2
        x_min = 1.5 - math.sqrt(2)
        vmax = (x_max - m) * x_max**0.25 * math.exp(-0.5 * x_max)
    else:
        vmax = 0
        x_min = m/2 - math.sqrt(4*m*m + 12*m + 25)/4. + 1.25

    umax = m**0.25 * math.exp(-0.5 * m)
    vmin = (x_min - m) * x_min**0.25 * math.exp(-0.5 * x_min)
    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(size, dtype='float64')

    with x.lock, nogil:
        for i in range(size):
            while (1):
                u = rng.next_double(rng.state) * umax
                v = vmin + rng.next_double(rng.state) * (vmax - vmin)
                z = v / u + m
                if 0 < z < chi2 / 2:
                    if u * u <= math.sqrt(z) * math.exp(-z):
                        break
            random_values[i] = math.sqrt(1 - 2 * z / chi2)

    return np.asarray(random_values)