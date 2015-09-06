from setuptools import setup

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = cythonize(

           "nula/functions/lstm_cpu_funcs.pyx",      # our Cython source
           #language="c++",             # generate C++ code
           #sources=["*.cpp"],  # additional source file(s)
           include_path = [np.get_include()],
      )
      
ext_modules += cythonize(
           "nula/cpu/utils.pyx",      # our Cython source
           #language="c++",             # generate C++ code
           #sources=["*.cpp"],  # additional source file(s)
           include_path = [np.get_include()],
)

extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)


setup(
    name='nula',
    packages=['nula',
              'nula.functions',
              'nula.models',
              'nula.gpu',
              'nula.cpu'],
    #package_data = {
     #   'nula': ['cuda_code.cu']    
    #},
    ext_modules = ext_modules,
    include_dirs = [np.get_include()],
    cmdclass = {'build_ext': build_ext},
    install_requires=['chainer>=1.3.0', 'numpy>=1.9',
                      'cython>=0.22']
    )