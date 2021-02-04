# Generating ARGUS random variates

This repository contains the implementations to generate random variates of the ARGUS distribution (https://en.wikipedia.org/wiki/ARGUS_distribution).

The implementation of the fastest method relies on the C library UNU.RAN that needs to be installed to use it. 

## Installation of UNU.RAN

Instructions can be found here. To use it with Cython, make sure that the compiler flag `-fPIC` is enabled. A few more details on my installation can be found below:

- UNU.RAN is installed in the subfolder `./code/unuran`
  - I used `sh ./configure --prefix=/home/.../code/unuran/ --enable-shared --with-pic`
  - in the makefile, edit the variable `CFLAGS`: `CFLAGS = -fPIC ...`, then run `make` and `make install`
- `libunuran.a` is in `/home/christoph/Documents/argus/unuran/lib`
- Ubuntu 20.04
- gcc (Homebrew GCC 5.5.0_7) 5.5.0

## Wrapping a C library with Cython

Good resources can be found here:

- https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html
- https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html
- https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/

To translate the file `unuran.h` into a `pxd` file, I used `autopxd` (use `pip install autopxd2`, see https://github.com/gabrieldemarmiesse/python-autopxd2). The most important objects in UNU.RAN are

```
struct unur_distr;                       
typedef struct unur_distr UNUR_DISTR;
struct unur_par;                         
typedef struct unur_par   UNUR_PAR;
struct unur_gen;                         
typedef struct unur_gen   UNUR_GEN;
```

They are described here: http://statmath.wu.ac.at/software/unuran/doc/unuran.html#Concepts

## Information on the repository

- *code* contains Python and Cython code to generate ARGUS random variates, code used to analyze the performance and to check the histograms of the samples against the density . To use the Cython code, you need to have Cython installed and to compile the pyx-file (`python setup.py build_ext --inplace`). If UNU.RAN should be use, you need to link the library before compiling the code by setting the environment variable `$LD_LIBRARY_PATH` by entering `export LD_LIBRARY_PATH=/home/.../code/unuran/lib:/usr/lib` in your command line
- *tables* contains the results of the performance analysis
- the file `env.yml` contains the Python environment used for all analysis / testing
