
import numpy as np
from chainer import cuda
from nula import function
from chainer.utils import type_check

if cuda.available:
    import cupy as cp
    from cupy.random import generator
    from cupy.cuda import curand
    from nula import gpu
    
class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio, no_cols):
        self.dropout_ratio = dropout_ratio
        self.no_cols = no_cols
        self.y = None

    def prepare(self, batchsize, on_gpu):
        xp = cp if on_gpu else np
        self.mask = xp.empty((batchsize, self.no_cols), dtype=np.dtype('float32'))
        self.y    = xp.empty_like(self.mask)
        self.gx   = xp.empty_like(self.mask)

    def check_type_forward(self, in_types):
        x, = in_types
        type_check.expect(
            x.shape[1] == self.no_cols,
            x.dtype == np.float32,
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        scale = 1. / (1 - self.dropout_ratio)
        N = x.shape[0]
        if self.y is None or self.y.shape[0] != N or type(self.y) != type(x):
            self.prepare(N, (xp == cp) )
            
        if xp is np:
            self.mask = np.random.rand(*x[0].shape)
            self.mask = scale * (self.mask >= self.dropout_ratio)
            np.multiply(self.mask, x, out=self.y)
        else:
            _get_mask_and_apply_dropout(x, self.mask, self.y, self.dropout_ratio, scale)
        return self.y,
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        gy, = grad_outputs
        if xp is np:
            self.gx = np.multiply(gy, self.mask, out=self.gx)
        else:
            gpu.utils.hadamard(gy, self.mask, out=self.gx)
        return self.gx,

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

