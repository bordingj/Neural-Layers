
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula.functions import add_matvec_elementwise_prod_cpu_funcs

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
        ga = xp.zeros_like(a)
        if xp is np:
            gx = a*gy
            ga = add_matvec_elementwise_prod_cpu_funcs.get_ga_cpu(gy, x, ga)
        else:
            gx=self.y
            gx = _MatVecElementwise_gpu(gy, a, gx)
            ga = _get_ga_gpu(gy, x, ga)
            
        return ga, gx, gc
        
def addMatVecElementwiseProd(a, x, c):
    return AddMatVecElementwiseProd()(a, x, c)

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

@cp.util.memoize(for_each_device=True)
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
    
    _MatVecElementwise = _get_MatVecElementwise_kernel()
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

@cp.util.memoize(for_each_device=True)
def _get_ga_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void get_ga(const float *gy, const float *x, float *ga, 
                                  const int N, const int M)
    {   
        __shared__ float shared_mat[16];
        float sum = 0;
        
        if (blockIdx.x < N){
        for (unsigned int i = threadIdx.x; i < M; i += 16) {
            sum += gy[blockIdx.x * M + i]*x[blockIdx.x * M + i];
        }
        shared_mat[threadIdx.x] = sum;
        __syncthreads();
        
    
        if (threadIdx.x == 0) {
            float total_sum = 0;
            
            for (unsigned int i = 0; i < 16; i++){
                total_sum += shared_mat[i];           
            }
            ga[blockIdx.x] = total_sum;
            
        }
    }
    }
    """)
    return kernel_code.get_function('get_ga')


def _get_ga_gpu(gy, x, ga):
    
    
    _get_ga = _get_ga_kernel()
    N, M = x.shape

    _get_ga(grid=(N, 1, 1), block=(32, 1, 1),
             args=(gy, x, ga,
                   np.int32(N),
                   np.int32(M)
                   ) )
    return ga
