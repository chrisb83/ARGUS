# Generating ARGUS random variates

This repository contains various algorithms to generate random variates of the ARGUS distribution (https://en.wikipedia.org/wiki/ARGUS_distribution).

## Information on the repository

- `code_article` contains Python and Cython code that was used to prepare the article [1]. It contains functions to generate ARGUS random variates, to analyze the performance and to check the histograms of the samples against the density. To use the Cython code, you have UNU.RAN installed (see the next section) and Cython is required to compile the pyx-file (`python setup.py build_ext --inplace`). If UNU.RAN should be used (note that the algorithms relying on the Ratio-of-Uniforms method do not need UNU.RAN), you need to link the library before compiling the code by setting the environment variable `$LD_LIBRARY_PATH` by entering `export LD_LIBRARY_PATH=/home/Documents/unuran/lib:/usr/lib` in your command line (adjust the paths to align with your installation path of UNU.RAN as described in the next section). It is not recommend to work with this code, instead use the one in the folder `code_simplified`, see the next point.
- `code_simplified` contains the main algorithms.
  - The easiest approach is to use the Python code in `main_algos.py`. For the inversion algorithms (Algorithms 1 and 2), it relies on SciPy >= 1.8.0 that includes UNU.RAN. The RoU algorithm (Algorithm 3) also works with older versions of SciPy.
  - The Cython code in `main_algos_cython.pyx` contains the same algorithms implemented in Cython. As before, it requires installing UNU.RAN and compilation of the Cython code (see the guidance above).
- `tables` contains the results of the performance analysis with the code in `code_article`
- the file `env.yml` contains the Python environment used for all analysis / testing of the code in `code_article`

## Installation of UNU.RAN

Instructions can be found on the page https://statmath.wu.ac.at/unuran/doc/unuran.html#Installation. To use it with Cython, make sure that the compiler flag `-fPIC` is enabled. A few more details on the installation (tested on Ubuntu 20.04 with gcc 5.5 (Homebrew GCC 5.5.0_7)) can be found below:

- After cloning the ARGUS repo (here, we refer to the location `/home/Documents/ARGUS/`), install UNU.RAN into a subfolder `./unuran`
  - After downloading UNU.RAN, unzip it and open a shell in the folder and run `sh ./configure --prefix=/home/Documents/unuran/ --enable-shared --with-pic`
  - in the makefile, edit the variable `CFLAGS`: `CFLAGS = -fPIC ...`, then run `make` and `make install`
- `libunuran.a` is in `/home/Documents/ARGUS/unuran/lib`
## Wrapping a C library with Cython

Good resources can be found here:

- https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html
- https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html
- https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/

To translate the file `unuran.h` into a `pxd` file, we used `autopxd` (`pip install autopxd2`, see https://github.com/gabrieldemarmiesse/python-autopxd2). The generated file `cpinv.pxd` is part of the repository, so this step does not have to be run by a user.

The most important objects in UNU.RAN are

```
struct unur_distr;
typedef struct unur_distr UNUR_DISTR;
struct unur_par;
typedef struct unur_par   UNUR_PAR;
struct unur_gen;
typedef struct unur_gen   UNUR_GEN;
```

They are described here: http://statmath.wu.ac.at/software/unuran/doc/unuran.html#Concepts

# References

[1] Christoph Baumgarten: Random Variate Generation by Fast Numerical Inversion in the Varying Parameter Case. Preprint.
