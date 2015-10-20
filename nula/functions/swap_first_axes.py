
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula.functions import swap_first_axes_cpu_funcs

from nula import cpu

if cuda.available:
    import cupy as cp
    from nula import gpu

    @cp.util.memoize(for_each_device=True)
    def _GetForward_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void forward(const float* x, float* y, const int T, const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
            
                if (i<N && j<M){
                    int p;
                    int q;
                    for (int t=0; t<T; t++){
                            y[i*(T*M) + t*M + j] = x[t*(N*M) + i*M + j];
                    }
                }
            }
            """)
        return kernel_code.get_function('forward')
        

    @cp.util.memoize(for_each_device=True)
    def _GetBackward_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void backward(const float* gy, float* gx, const int T, const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
            
                if (i<N && j<M){
                    int p;
                    int q;
                    for (int t=0; t<T; t++){
                            gx[t*(N*M) + i*M + j] = gy[i*(T*M) + t*M + j];
                    }
                }
            }
            """)
        return kernel_code.get_function('backward')



def _forward_gpu(x):
    
    T = x.shape[0]
    N = x.shape[1]
    M = x.shape[2]
    y = cp.empty((N, T, M), dtype=np.float32)
    
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    forward_kernel = _GetForward_kernel()
    
    forward_kernel(grid=gdim, block=bdim,
                   args=(x, y,
                         T, N, M
                         )
                    )  
    return y

def _backward_gpu(gy):
    
    N = gy.shape[0]
    T = gy.shape[1]
    M = gy.shape[2]
    gx = cp.empty((T, N, M), dtype=np.float32)
    
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    

    Backward_kernel = _GetBackward_kernel()
    
    
    Backward_kernel(grid=gdim, block=bdim,
                   args=(gy, gx,
                         T, N, M)
                    )  
    return gx


class SwapFirstAxes(function.Function):
    
    def check_type_forward(self, in_types):
        x = in_types[0]
        type_check.expect(
            x.dtype == np.float32,
            x.ndim == 3
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        
        if xp is np:
            y = swap_first_axes_cpu_funcs.forward(x)
        else:
            y = _forward_gpu(x)
        return y,
        
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gy = grad_outputs[0]

        if xp is np:
            gx = swap_first_axes_cpu_funcs.backward(gy)
        else:
            gx = _backward_gpu(gy)
        return gx,
        
def swapfirstaxes(x):
    return SwapFirstAxes()(x)