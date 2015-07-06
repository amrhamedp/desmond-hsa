from distutils.core import setup, Extension
import numpy
# define the extension module
hsa_c_module = Extension('_hsacalcs_v2', sources=['_hsacalcs_v2.c'], include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[hsa_c_module])