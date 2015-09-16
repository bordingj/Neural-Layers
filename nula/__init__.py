
from nula import cpu

from chainer import cuda
if cuda.available:
    from nula import gpu

from nula import functions

from nula import models
