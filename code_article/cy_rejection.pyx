# -----------------------------------------------------------------------------
# implementation of ARGUS generators - rejection methods
# -----------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from libc cimport math
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64


# -----------------------------------------------------------------------------
# rejection from the beta density
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def rvs_rejection_beta_cy(double chi, Py_ssize_t size=1):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, z, d
    cdef double[::1] random_values

    # note: the function works with chi = 0 (immediate acceptance)
    if chi < 0:
        return np.empty(size, dtype='float64') * np.nan
    
    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(size, dtype='float64')

    d = -0.5 * chi * chi
    with x.lock, nogil:
        for i in range(size):
            while (1):
                u = rng.next_double(rng.state)
                v = rng.next_double(rng.state)
                z = v**(2/3)
                if math.log(u) <= d * z:
                    break
            random_values[i] = math.sqrt(1 - z)

    return np.asarray(random_values)


# -----------------------------------------------------------------------------
# rejection from x * exp(...)
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_rejection_xexp_cy(double chi, Py_ssize_t size=1):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, z, chi2, echi
    cdef double[::1] random_values
    
    if chi <= 0:
        return np.empty(size, dtype='float64') * np.nan

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(size, dtype='float64')

    echi = math.exp(-0.5 * chi * chi)
    chi2 = chi * chi
    with x.lock, nogil:
        for i in range(size):
            while (1):
                u = rng.next_double(rng.state)
                v = rng.next_double(rng.state)
                # division by zero not possible as nan is returned above
                z = 2 * math.log(echi * (1 - v) + v) / chi2
                if u*u + z <= 0:
                    break
            random_values[i] = math.sqrt(1 + z)

    return np.asarray(random_values)
