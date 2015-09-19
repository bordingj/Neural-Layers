# Neural-Layers: nula - Chainer-based framework for neural networks
nula is a small package with a set of chainer/nula-functions which can be used to build neural networks in python. These functions are available in nula/functions. nula also provides some prebuild models available in nula/models.

Dependencies:
 - Python 3.4+
 - Chainer 1.3+
 - cython 0.22+
 - numpy 1.9+

nula reimplements some functions which are already available in chainer, but some of these reimplementations can be more efficient and use somewhat more efficient cuda-code (using raw cuda-code instead of elementwise-kernels). 

