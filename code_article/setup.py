from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="cy_rou_gamma",
        sources=["cy_rou_gamma.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension(
        name="cy_rou_maxwell",
        sources=["cy_rou_maxwell.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension(
        name="cy_rejection",
        sources=["cy_rejection.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension(
        name="cy_pinv",
        sources=["cy_pinv.pyx"],
        libraries=["unuran"],
        library_dirs=["../unuran/lib"],
        include_dirs=["../unuran/include",
                      numpy.get_include()]),
    Extension(
        name="test_varying",
        sources=["test_varying.pyx"],
        libraries=["unuran"],
        library_dirs=["../unuran/lib"],
        include_dirs=["../unuran/include",
                      numpy.get_include()])
]

setup(ext_modules=cythonize(ext_modules, language_level="3", annotate=True))