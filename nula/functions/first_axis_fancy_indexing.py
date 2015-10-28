
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula.functions import first_axis_fancy_indexing_cpu_funcs

if cuda.available:
    import cupy as cp
    from nula import gpu
    
    @cp.util.memoize(for_each_device=True)
    def _GetForward_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void forward(const float* x, float* y, const int* indices, const int offset,
                         const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
            
                if (i<N && j<M){
                    int t = indices[i]+offset;
                    y[i*M+j] = x[t*N*M+i*M+j];
                }
            }
            """)
        return kernel_code.get_function('forward')
    
    @cp.util.memoize(for_each_device=True)
    def _GetBackward_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void backward(float* gx, const float* gy, const int* indices, const int offset,
                         const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
            
                if (i<N && j<M){
                    int t = indices[i]+offset;
                    gx[t*N*M+i*M+j] = gy[i*M+j];
                }
            }
            """)
        return kernel_code.get_function('backward')


def _forward_gpu(x, y, indices, offset):
    
    N = y.shape[0]
    M = y.shape[1]
    
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    forward_kernel = _GetForward_kernel()
    
    forward_kernel(grid=gdim, block=bdim,
                   args=(x, y, indices,
                         offset, N, M
                         )
                    )
    return y

def _backward_gpu(gx, gy, indices, offset):
    
    N = gy.shape[0]
    M = gy.shape[1]
    
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    backward_kernel = _GetBackward_kernel()
    
    backward_kernel(grid=gdim, block=bdim,
                   args=(gx, gy, indices,
                         offset, N, M
                         )
                    )
    return gx

class FirstAxisFancyIndexing3D(function.Function):
    

    def __init__(self, offset=0):
        self.offset = offset

    def check_type_forward(self, in_types):
        x, indices = in_types
        type_check.expect(
            x.dtype == np.float32,
            indices.dtype == np.int32,
            indices.ndim == 1,
            x.ndim == 3,
            x.shape[1] == indices.shape[0],
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, indices = inputs
        N, D = x.shape[1:]
        
        y = xp.empty((N, D), dtype=np.float32)
        if xp is np:
            y = first_axis_fancy_indexing_cpu_funcs.forward(x, y, indices, self.offset)
        else:
            y = _forward_gpu(x, y, indices, self.offset)
        return y,
        
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, indices = inputs
        gy = grad_outputs[0]
        
        gx   = xp.zeros_like(x)
        
        if xp is np:
            gx = first_axis_fancy_indexing_cpu_funcs.backward(gx, gy, indices, self.offset)
        else:
            gx = _backward_gpu(gx, gy, indices, self.offset)
            
        return gx, None

def firstAxisFancyIndexing3D(x, indices, offset=0):
    return FirstAxisFancyIndexing3D(offset)(x, indices)