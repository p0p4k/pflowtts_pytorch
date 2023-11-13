from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup

# exts = [
#     Extension(
#         name="pflow.utils.monotonic_align.core",
#         sources=["pflow/utils/monotonic_align/core.pyx"],
#     )
# ]
# setup(name='monotonic_align',
#       ext_modules=cythonize(exts, language_level=3),
#       include_dirs=[numpy.get_include()])

setup(
    name="monotonic_align",
    ext_modules=cythonize("pflow/utils/monotonic_align/core.pyx"),
    include_dirs=[numpy.get_include()],
)
