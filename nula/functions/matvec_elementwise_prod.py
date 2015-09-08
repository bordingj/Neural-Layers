
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
import math

if cuda.available:
    import cupy as cp
    from nula import gpu
    
class AddMatVecElementwiseProd(function.Function):

    """Matrix-vector elementwise product with broadcasting and adding:
    y = a*x+c, where a is a vector, x is a matrix and c is a matrix"""

    def check_type_forward(self, in_types):
        a, x, c = in_types
        type_check.expect(
            x.ndim == 2,
            a.ndim == 2,
            c.shape == x.shape,
            a.shape[1] == 1,
            a.shape[0] == x.shape[0],
            x.dtype == np.float32,
            c.dtype == a.dtype,
            x.dtype == c.dtype
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        a, x, c = inputs   
            
        if xp is np:
            self.y = a*x+c
        else:
            y = xp.empty_like(x)
            self.y = _forward_gpu(x, c, a, y)
        return self.y,
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        a, x, c = inputs
        gy, = grad_outputs
        
        gc = gy
        ga = xp.empty_like(a)
        if xp is np:
            gx = a*gy
            ga = _get_ga_cpu(gy, x, ga)
        else:
            gx = _MatVecElementwise_gpu(gy, a, gx=self.y)
            
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
def _get_forward_kernel():
    
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void AddMatVecElementwiseProd(const float *x, const float *c, 
                                  const float *a, float *y, 
                                  const int N, const int M)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
    
            if (i < N && j < M){
            y[i*M+j] = a[i]*x[i*M+j] + c[i*M+j];
        }
    }
    """)
    return kernel_code.get_function('AddMatVecElementwiseProd')

def _forward_gpu(x, c, a, y):
    
    
    _Forward = _get_forward_kernel()
    N, M = x.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M) 
    
    _Forward(grid=gdim, block=bdim,
             args=(x, c, a, y,
                   np.int32(N),
                   np.int32(M)
                   ) )
    return y

def _get_MatVecElementwise_kernel():
    
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void MatVecElementwiseProd(const float *gy, const float *a, float *gx, 
                                  const int N, const int M)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
    
            if (i < N && j < M){
            gx[i*M+j] = a[i]*gy[i*M+j];
        }
    }
    """)
    return kernel_code.get_function('MatVecElementwiseProd')

def _MatVecElementwise_gpu(gy, a, gx):
    
    _MatVecElementwise = _get_forward_kernel()
    N, M = gy.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M) 
    
    _MatVecElementwise(grid=gdim, block=bdim,
             args=(gy, a, gx,
                   np.int32(N),
                   np.int32(M)
                   ) )
    return gx

def _get_ga_cpu(gy, x, ga):
    N, M = gy.shape    
    for i in range(N):
        ga[i] = 0
        for j in range(M):
            ga[i] += gy[i,j]*x[i,j]
    return ga

def _get_ga_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void get_ga(const float *gy, const float *x, float *ga, 
                                  const int N, const int M)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        extern __shared__ float shared_mat[N*8] = {0};
        
            if (i < N){
            for (int k=0; k<(M+8-1)/8; k++){
                int idx = i*M+j*8+k;
                if {idx < M}{
                    shared_mat[i*M+j] += gy[idx]*x[idx];
                }
            }
            __syncthreads();
            
            if {j == 0}{
                ga[i] = 0;
                for (int k=0;k<8;k++){
                    ga[i] += shared_mat[i*M+j];     
                }
            }
        }
    }
    """)
    return kernel_code.get_function('get_ga')

def _get_ga_gpu(gy, x, ga):
    
    
    _get_ga = _get_ga_kernel()
    N, M = x.shape
    num_threads_cols = math.ceil(M/8)
    num_threads_rows = 512//num_threads_cols
    bdim = (num_threads_rows, num_threads_cols, 1)
    gdim = (math.ceil(num_threads_rows/N),1,1)
    
    _get_ga(grid=gdim, block=bdim,
             args=(gy, x, ga,
                   np.int32(N),
                   np.int32(M)
                   ) )
    return ga
