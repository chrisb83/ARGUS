# -----------------------------------------------------------------------------
# ARGUS generators - Ratio-of-Uniforms method based on Gamma distribution
# -----------------------------------------------------------------------------

import numpy as np
cimport numpy as np
from libc cimport math
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms with mode shift
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


# -----------------------------------------------------------------------------
# Ratio-of-Uniforms without mode shift
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_rou_gamma_cy(double chi, Py_ssize_t size=1):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmax, umax, z, chi2, m
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

    if chi2 >= 5:
        vmax = 2.5**(1.25) * math.exp(-1.25)
    else:
        vmax = chi**2.5 * math.exp(-0.25 * chi2) / 2**1.25

    umax = m**(0.25) * math.exp(-m / 2)

    with x.lock, nogil:
        for i in range(size):
            while (1):
                u = rng.next_double(rng.state) * umax
                v = rng.next_double(rng.state) * vmax
                z = v / u
                if z < chi2 / 2:
                    if u * u <= math.sqrt(z) * math.exp(-z):
                        break
            random_values[i] = math.sqrt(1 - 2 * z / chi2)

    return np.asarray(random_values)
