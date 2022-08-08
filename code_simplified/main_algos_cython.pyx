# -----------------------------------------------------------------------------
# implementation of ARGUS generators - numerical inversion
# -----------------------------------------------------------------------------
import numpy as np
cimport numpy as np
cimport cpinv
cimport cython
from libc cimport math
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from scipy.special cimport cython_special


# -----------------------------------------------------------------------------
# Algorithm 1: Gamma density (suggested for chi > 5)
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
# Algorithm 1: ARGUS density (suggested for chi < 5)
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
# Algorithm 2
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
    def rvs_varying(self, params):
        cdef double u, ub, c, y, chi, v, sqrt2pi
        cdef Py_ssize_t i, N
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values
        cdef double[::1] params_1d

        params_arr = np.array(params)
        params_1d = params_arr.ravel()

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        N = len(params_1d)
        random_values = np.empty(N, dtype='float64')

        v = 0.0
        sqrt2pi = math.sqrt(2.0/math.M_PI)
        # generate ARGUS(chi) rvs by transforming conditioned Gamma rvs
        for i in range(N):
            chi = params_1d[i]
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

        return np.asarray(random_values).reshape(params_arr.shape)


# -----------------------------------------------------------------------------
# Algorithm 3: Ratio-of-Uniforms with mode shift
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_rou_shifted_gamma_cy(double chi, Py_ssize_t size=1):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmin, vmax, umax, z, chi2, m, x_min, x_max
    cdef double[::1] random_values

    if chi <= 0:
        raise ValueError('chi must be > 0')

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(size, dtype='float64')

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
