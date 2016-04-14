from distutils.core import setup, Extension
import numpy

c_extension_module = Extension('_hsacalcs', sources=['_hsacalcs.c'], include_dirs=[numpy.get_include()])

setup(ext_modules=[c_extension_module])

