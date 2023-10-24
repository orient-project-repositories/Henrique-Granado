from distutils.core import setup, Extension
from Cython.Build import cythonize

# setup(ext_modules=cythonize('./cy_utils.pyx'))

# EXT_MODULES = cythonize([
#     Extension("System", ["./System.pyx"],
#               extra_compile_args=["-Ox", "-Zi"],
#               extra_link_args=["-debug:full"])
# ], emit_linenums=True)
#
# setup(ext_modules= EXT_MODULES)

setup(ext_modules=cythonize('./System.pyx'))