
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula.functions import pause_mask_cpu_funcs

if cuda.available:
    import cupy as cp
    from nula import gpu

    @cp.util.memoize(for_each_device=True)
    def _get_mask_kernel():
        
        kernel_code = cp.carray.compile_with_cache("""
        extern "C" __global__
        void get_mask(const int* indices, const int* pause_indices, 
                float* row_masks, const int N, const int no_pause_indices)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
        
            if (i < N){
                int was_a_pause = 0;
                for (int k=0; k<no_pause_indices; k++){
                    if (indices[i] == pause_indices[k]){
                        was_a_pause = 1;
                        break;
                    }
                }
                if (was_a_pause){
                    row_masks[i] = 1.0;
                }else{
                    row_masks[i] = 0.0;            
                }
            }
        }
        """)
        return kernel_code.get_function('get_mask')
    
    @cp.util.memoize(for_each_device=True)
    def _get_vecMatElemProd_kernel():
        
        kernel_code = cp.carray.compile_with_cache("""
        extern "C" __global__
        void vecMatElemProd(const float* x, const float* row_masks, float* y, 
                            const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
        
            if (i < N && j < M){
                y[i*M+j] = row_masks[i]*x[i*M+j];
            }
        }
        """)
        return kernel_code.get_function('vecMatElemProd')
        
class PauseMask(function.Function):

    def __init__(self, pause_indices):
        self.pause_indices = pause_indices


    def check_type_forward(self, in_types):
        x, indices = in_types
        type_check.expect(
            x.dtype == np.float32,
            indices.dtype == np.int32,
            indices.ndim == 1,
            x.ndim == 2,
            x.shape[0] == indices.shape[0]
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, indices = inputs
        
        self.row_masks = xp.empty((x.shape[0],), dtype=np.float32)
        y             = xp.empty_like(x)
        
        if xp is np:
            y = pause_mask_cpu_funcs.forward(x, indices, self.pause_indices, self.row_masks, y)
        else:
            y = _forward_gpu(x, indices, self.pause_indices, self.row_masks, y)
        return y,
        
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, indices = inputs
        gy = grad_outputs[0]
        
        gindices = None
        
        gx   = xp.empty_like(x)
        
        if xp is np:
            gx = np.multiply(gy, self.row_masks, out=gx)
        else:
            gx = _backward_gpu(gy, self.row_masks, gx)
        return gx, gindices
        
def pauseMask(x, indices, pause_indices):
    return PauseMask(pause_indices)(x, indices)


    
def _forward_gpu(x, indices, pause_indices, row_masks, y):

    N, M = x.shape
    no_pause_indices = pause_indices.shape[0]
    
    bdim, gdim = gpu.utils.Get_bdim_and_gdim1D(N)
    
    Get_mask_kernel = _get_mask_kernel()
    
    Get_mask_kernel(grid=gdim, block=bdim,
                    args=(indices, pause_indices,
                          row_masks, np.int32(N), np.int32(no_pause_indices)))
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    vecMatElemProd_kernel = _get_vecMatElemProd_kernel()
    
    vecMatElemProd_kernel(grid=gdim, block=bdim,
                          args=(x, row_masks, y,
                                np.int32(N), np.int32(M))
                                )
    return y

def _backward_gpu(gy, row_masks, gx):
    
    N, M = gy.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    vecMatElemProd_kernel = _get_vecMatElemProd_kernel()
    
    vecMatElemProd_kernel(grid=gdim, block=bdim,
                          args=(gy, row_masks, gx,
                                np.int32(N), np.int32(M))
                                )
    return gx
