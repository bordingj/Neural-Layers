
import numpy as np
from chainer import cuda
from chainer import function

if cuda.available:
    import cupy as cp
    from nula import gpu
    
class ConvexComb1d(function.Function):

    def __init__(self):
        self.alpha = np.array([0.5],dtype=np.float32)
        self.galpha = np.empty_like(self.alpha)

    @property
    def parameter_names(self):
        return 'alpha',
 
    @property
    def gradient_names(self):
        return 'galpha',
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        a, b= inputs
        c = self.alpha[0]*a+(xp.float32(1.0)-self.alpha[0])*b
        return c,
        
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        a, b= inputs
        gc = grad_outputs[0]
        self.galpha += gc*(a-b)
        
        ga = gc*self.alpha[0]
        gb = gc*(xp.float32(1.0)-self.alpha[0])
        return ga, gb