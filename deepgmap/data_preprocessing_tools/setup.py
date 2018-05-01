from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "seq_to_binary2",
    ext_modules = cythonize('seq_to_binary2.pyx'),  # accepts a glob pattern
)
