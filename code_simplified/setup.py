from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="argus_cy",
        sources=["main_algos_cython.pyx"],
        libraries=["unuran"],
        library_dirs=["../unuran/lib"],
        include_dirs=["../unuran/include",
                      numpy.get_include()])
]

setup(ext_modules=cythonize(ext_modules, language_level="3", annotate=True))