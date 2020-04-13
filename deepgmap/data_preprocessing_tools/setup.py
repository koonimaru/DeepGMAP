from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "seq_to_binary",
    ext_modules = cythonize('seq_to_binary.pyx'),  # accepts a glob pattern
)
