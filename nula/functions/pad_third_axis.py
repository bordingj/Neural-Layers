
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.available:
    import cupy as cp
    from nula import gpu


class PadThirdAxis(function.Function):
    
    def __init__(self, pad_width):
        self.pad_width = pad_width
    
    def check_type_forward(self, in_types):
        x = in_types[0]
        type_check.expect(
            x.dtype == np.float32,
            x.ndim == 4
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        y = xp.zeros(x.shape[:2]+(x.shape[2]+self.pad_width*2,)+(x.shape[3],),dtype=np.float32)
        y[:,:,self.pad_width:(y.shape[2]-self.pad_width),:] = x
        return y,
        
    
    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        gx = gy[:,:,self.pad_width:(gy.shape[2]-self.pad_width),:]
        return gx,
        
def padthirdaxis(x, pad_width):
    return PadThirdAxis(pad_width)(x)