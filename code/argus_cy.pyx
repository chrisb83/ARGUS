import numpy as np
cimport numpy as np
from libc cimport math
from numpy.math cimport PI
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64

@cython.boundscheck(False)
@cython.wraparound(False)
def rvs_rou_shifted_gamma_cy(double chi, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmin, vmax, umax, z, chi2, m, x_min, x_max
    cdef double[::1] random_values

    if chi <= 0:
        return np.empty(n, dtype='float64') * np.nan

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')
    
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
        for i in range(n):
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
def rvs_rejection_xexp_cy(double chi, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, z, chi2, echi
    cdef double[::1] random_values
    
    if chi <= 0:
        return np.empty(n, dtype='float64') * np.nan

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')

    echi = math.exp(-0.5 * chi * chi)
    chi2 = chi * chi
    with x.lock, nogil:
        for i in range(n):
            while (1):
                u = rng.next_double(rng.state)
                v = rng.next_double(rng.state)
                # division by zero not possible as nan is returned above
                z = 2 * math.log(echi * (1 - v) + v) / chi2
                if u*u + z <= 0:
                    break
            random_values[i] = math.sqrt(1 + z)

    return np.asarray(random_values)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_rou_maxwell_cy(double chi, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmax, umax, z, z2, chi2
    cdef double[::1] random_values

    if chi <= 0:
        return np.empty(n, dtype='float64') * np.nan

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')
    
    chi2 = chi * chi
    
    if chi2 < 2:
        umax = chi * math.exp(-chi2 / 4)
        vmax = chi * umax
    else:
        umax = math.sqrt(2) / math.exp(0.5)
        vmax = 4 / math.exp(1)
    
    # note: division by zero cannot occur in the loop
    with x.lock, nogil:
        for i in range(n):
            while (1):
                u = rng.next_double(rng.state) * umax
                v = rng.next_double(rng.state) * vmax
                z = v / u
                if z < chi:
                    z2 = z * z
                    if u <= z * math.exp(-0.25 * z2):
                        break
            random_values[i] = math.sqrt(1 - z2 / chi2)

    return np.asarray(random_values)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_rou_shifted_maxwell_cy(double chi, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, u2, v2, vmin, vmax, umax, z, m, chi2, x0, x1, p, q, c1, c2
    cdef double[::1] random_values

    if chi <= 0:
        return np.empty(n, dtype='float64') * np.nan

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')

    chi2 = chi * chi
    if chi2 <= 2:
        m = chi
    else:
        m = math.sqrt(2)

    p = -4 - m**2 / 3
    q = -2*m**3/27 - 4*m/3 + 2*m
    c1 = math.sqrt(-p / 3)
    # note: c1 > sqrt(4/3), p < -4, so division by 0 cannot occur
    c2 = math.acos(1.5*q/p/c1)/3
    x1 = 2*c1*math.cos(c2 - 2*PI/3) + m/3
    
    if chi2 <= 2:
        vmax = 0
    else:
        x0 = 2*c1*math.cos(c2) + m/3
        if x0 < chi:
            vmax = (x0 - m) * x0 * math.exp(-x0*x0 / 4)
        else:
            vmax = (chi - m) * chi * math.exp(-chi2 / 4)
    
    umax = m * math.exp(-m*m / 4)
    vmin = (x1 - m) * x1 * math.exp(-x1**2 / 4)
    
    with x.lock, nogil:
        for i in range(n):
            while (1):
                u = rng.next_double(rng.state) * umax
                v = vmin + rng.next_double(rng.state) * (vmax - vmin)
                z = v / u + m
                if 0 < z < chi:
                    if u <= z * math.exp(-0.25 * z * z):
                        break
            # division by zero not possible since chi2 > 0
            random_values[i] = math.sqrt(1 - z * z / chi2)

    return np.asarray(random_values)


@cython.boundscheck(False)
@cython.wraparound(False)
def rvs_rejection_beta_cy(double chi, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, z, d
    cdef double[::1] random_values

    # note: the function works with chi = 0 (immediate acceptance)
    if chi < 0:
        return np.empty(n, dtype='float64') * np.nan
    
    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')

    d = -0.5 * chi * chi
    with x.lock, nogil:
        for i in range(n):
            while (1):
                u = rng.next_double(rng.state)
                v = rng.next_double(rng.state)
                z = v**(2/3)
                if math.log(u) <= d * z:
                    break
            random_values[i] = math.sqrt(1 - z)

    return np.asarray(random_values)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rvs_rou_gamma_cy(double chi, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double u, v, vmax, umax, z, chi2, m
    cdef double[::1] random_values

    if chi <= 0:
        return np.empty(n, dtype='float64') * np.nan

    x = PCG64()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n, dtype='float64')
    
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
        for i in range(n):
            while (1):
                u = rng.next_double(rng.state) * umax
                v = rng.next_double(rng.state) * vmax
                z = v / u
                if z < chi2 / 2:
                    if u * u <= math.sqrt(z) * math.exp(-z):
                        break
            random_values[i] = math.sqrt(1 - 2 * z / chi2)

    return np.asarray(random_values)

@cython.boundscheck(False)
@cython.wraparound(False)
def rvs_cond_gamma_cy(double chi, Py_ssize_t n):
    if chi <= 0:
        return np.empty(n, dtype='float64') * np.nan

    return 0.0
