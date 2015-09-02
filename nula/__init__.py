from chainer import cuda
from nula import cpu
if cuda.available:
    from nula import gpu
from nula import function
from nula import functions
from nula import models
