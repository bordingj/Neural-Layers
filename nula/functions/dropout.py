
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.available:
    import cupy as cp
    from cupy.random import generator
    from cupy.cuda import curand
    from nula import gpu
    
class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio


    def check_type_forward(self, in_types):
        x, = in_types
        type_check.expect(
            x.shape[1] == self.no_cols,
            x.dtype == np.float32,
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs   
        
        self.mask = xp.empty_like(x)
        y         = xp.empty_like(self.mask)
        
        scale = 1. / (1 - self.dropout_ratio)
            
        if xp is np:
            self.mask = np.random.rand(*x[0].shape)
            self.mask = scale * (self.mask >= self.dropout_ratio)
            np.multiply(self.mask, x, out=y)
        else:
            _get_mask_and_apply_dropout(x, self.mask, y, self.dropout_ratio, scale)
        return y,
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        gy, = grad_outputs
        
        gx   = xp.empty_like(self.mask)
        
        if xp is np:
            gx = np.multiply(gy, self.mask, out=gx)
        else:
            gpu.utils.hadamard(gy, self.mask, out=gx)
        return gx,

def dropout(x, ratio=.5, train=True):
    """Drops elements of input variable randomly.
    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.
    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.
        train (bool): If True, executes dropout. Otherwise, does nothing.
    Returns:
        ~chainer.Variable: Output variable.
    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_.
    """
    if train:
        return Dropout(ratio)(x)
    return x

@cp.util.memoize(for_each_device=True)
def _get_dropout_kernel():
    
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void Dropout(float* x, float* mask, float* y, float dropout_ratio,
                   float scale, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
    
            if (i < N){
            mask[i] = mask[i] < dropout_ratio ? 0 : scale;
            y[i]    = mask[i] * x[i];
        }
    }
    """)
    return kernel_code.get_function('Dropout')

def _get_mask_and_apply_dropout(x, mask, y, dropout_ratio, scale):
    
    
    _Dropout = _get_dropout_kernel()
    bdim, gdim = gpu.utils.Get_bdim_and_gdim1D(x.size)

    rs = generator.get_random_state()
    curand.generateUniform(rs._generator, mask.data.ptr, mask.size)
    
    _Dropout(grid=gdim, block=bdim,
             args=(x, mask, y,
                         np.float32(dropout_ratio),
                         np.float32(scale),
                         np.int32(x.size)
                   ) )
    return y, mask

