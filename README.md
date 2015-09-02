# Neural-Layers: nula - Chainer-based framework for neural networks
nula is a small package with a set of chainer/nula-functions which can be used to build neural networks in python. These functions are available in nula/functions. nula also provides some prebuild models available in nula/models.

Dependencies:
 - Python 3.4+
 - Chainer 1.3+
 - numba 0.20+
 - numpy 1.9+

nula reimplements some functions which are already available in chainer, but some of these reimplementations can be more efficient and use somewhat more efficient cuda-code. 

nula-functions are chainer-functions with an additional .copy() method whis performs a deep-copy of the arrays of the function except for its parameters and its gradients of its parameters. Moreover, nula-functions provides an option not to copy itself on a function-call (copy_func=..), thereby providing a possibility to reuse allocated arrays of the function (for example its output(s)). Note that YOU SHOULD NOT pass copy_func=False if the same instantiation of a function is called multiple times before running .backward() !!

It is perfectly fine to mix nula-functions and chainer-functions. A simple example of how to build a recurrent LSTM network can be found in nula/models/lstm_embed.py

Expect around a 10-30% speedup from using nula-functions instead of pure chainer.
