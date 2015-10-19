
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula.functions import extract_words_cpu_funcs

from nula import cpu

if cuda.available:
    import cupy as cp
    from nula import gpu

    @cp.util.memoize(for_each_device=True)
    def _GetForward_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void forward(const int* IDs, 
                         const float* x, float* y,
                         const int* no_whitespaces, const int* whitespace_IDs,
                         const int no_whitespace_ids, const int T, const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
            
                if (i<N && j<M){
                    int no_whitespaces_i = 0;
                    int n;
                    int p;
                    int q;
                    int current_max_whitespaces = no_whitespaces[i];
                    for (int t=0; t<T; t++){
                        if (no_whitespaces_i < current_max_whitespaces){
                            n = t*N+i*1+0;
                            q = no_whitespaces_i*(N*M)+i*M+j;
                            p = t*(N*M)+i*M+j;
                            for (int k=0; k<no_whitespace_ids; k++){
                                if (IDs[n] == whitespace_IDs[k]){
                                    y[q] = x[p];
                                    no_whitespaces_i += 1;
                                    break;
                                }
                            }
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            """)
        return kernel_code.get_function('forward')

    @cp.util.memoize(for_each_device=True)
    def _GetForward_axes_swapped_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void forward(const int* IDs, 
                         const float* x, float* y,
                         const int* no_whitespaces, const int* whitespace_IDs,
                         const int no_whitespace_ids, const int T, const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
            
                if (i<N && j<M){
                    int no_whitespaces_i = 0;
                    int n;
                    int p;
                    int q;
                    int current_max_whitespaces = no_whitespaces[i];
                    for (int t=0; t<T; t++){
                        if (no_whitespaces_i < current_max_whitespaces){
                            n = t*N+i*1+0;
                            q = i*(T*M)+no_whitespaces_i*M+j;
                            p = t*(N*M)+i*M+j;
                            for (int k=0; k<no_whitespace_ids; k++){
                                if (IDs[n] == whitespace_IDs[k]){
                                    y[q] = x[p];
                                    no_whitespaces_i += 1;
                                    break;
                                }
                            }
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            """)
        return kernel_code.get_function('forward')
        

    @cp.util.memoize(for_each_device=True)
    def _GetBackward_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void backward(const int* IDs, 
                         const float* gy, float* gx,
                         const int* no_whitespaces, const int* whitespace_IDs,
                         const int no_whitespace_ids, const int T, const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                
                if (i<N && j<M){
                    int no_whitespaces_i = 0;
                    int n;
                    int p;
                    int q;
                    int current_max_whitespaces = no_whitespaces[i];
                    for (int t=0; t<T; t++){
                        if (no_whitespaces_i < current_max_whitespaces){
                            n = t*N+i*1+0;
                            q = no_whitespaces_i*(N*M)+i*M+j;
                            p = t*(N*M)+i*M+j;
                            for (int k=0; k<no_whitespace_ids; k++){
                                if (IDs[n] == whitespace_IDs[k]){
                                    gx[p] = gy[q];
                                    no_whitespaces_i += 1;
                                    break;
                                }
                            }
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            """)
        return kernel_code.get_function('backward')

    @cp.util.memoize(for_each_device=True)
    def _GetBackward_axes_swapped_kernel():
        kernel_code = cp.carray.compile_with_cache("""
            extern "C" __global__
            void backward(const int* IDs, 
                         const float* gy, float* gx,
                         const int* no_whitespaces, const int* whitespace_IDs,
                         const int no_whitespace_ids, const int T, const int N, const int M){
                
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                
                if (i<N && j<M){
                    int no_whitespaces_i = 0;
                    int n;
                    int p;
                    int q;
                    int current_max_whitespaces = no_whitespaces[i];
                    for (int t=0; t<T; t++){
                        if (no_whitespaces_i < current_max_whitespaces){
                            n = t*N+i*1+0;
                            q = i*(T*M)+no_whitespaces_i*M+j;
                            p = t*(N*M)+i*M+j;
                            for (int k=0; k<no_whitespace_ids; k++){
                                if (IDs[n] == whitespace_IDs[k]){
                                    gx[p] = gy[q];
                                    no_whitespaces_i += 1;
                                    break;
                                }
                            }
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            """)
        return kernel_code.get_function('backward')
    

def _forward_gpu(IDs, x, y,
                no_whitespaces, whitespace_IDs,
                swapaxes):
    
    no_whitespace_ids = whitespace_IDs.shape[0]
    T = x.shape[0]
    N = x.shape[1]
    M = x.shape[2]
    
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    if swapaxes:
        forward_kernel = _GetForward_axes_swapped_kernel()
    else:
        forward_kernel = _GetForward_kernel()
    
    forward_kernel(grid=gdim, block=bdim,
                   args=(IDs, 
                         x, y,
                         no_whitespaces, whitespace_IDs,
                         no_whitespace_ids, T, N, M
                         )
                    )  
    return y

def _backward_gpu(IDs, gy, gx,
                no_whitespaces, whitespace_IDs,
                swapaxes):
    
    no_whitespace_ids = whitespace_IDs.shape[0]
    T = gx.shape[0]
    N = gx.shape[1]
    M = gx.shape[2]
    
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    if swapaxes:
        Backward_kernel = _GetBackward_axes_swapped_kernel()
    else:
        Backward_kernel = _GetBackward_kernel()
    
    
    Backward_kernel(grid=gdim, block=bdim,
                   args=(IDs, 
                         gy, gx,
                         no_whitespaces, whitespace_IDs,
                         no_whitespace_ids, T, N, M
                         )
                    )  
    return gx


class ExtractWords(function.Function):
    

    def __init__(self, no_whitespaces, whitespace_IDs, swapaxes=False):
        self.whitespace_IDs = whitespace_IDs
        self.no_whitespaces = no_whitespaces
        self.swapaxes = swapaxes

    def check_type_forward(self, in_types):
        x, IDs = in_types
        type_check.expect(
            x.dtype == np.float32,
            IDs.dtype == np.int32,
            IDs.ndim == x.ndim,
            x.ndim == 3,
            x.shape[0] == IDs.shape[0],
            x.shape[1] == IDs.shape[1],
            IDs.shape[2] == 1
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, IDs = inputs
        N, D = x.shape[1:]
        
        if xp is np:
            max_no_whitespaces = int(xp.amax(self.no_whitespaces))
            if self.swapaxes:
                y = xp.zeros((N, max_no_whitespaces, D), dtype=np.float32)
                y = extract_words_cpu_funcs.forward_axes_swapped(IDs, x, y, 
                                                        self.no_whitespaces, 
                                                        self.whitespace_IDs)
            else:
                y = xp.zeros((max_no_whitespaces, N, D), dtype=np.float32)
                y = extract_words_cpu_funcs.forward(IDs, x, y, 
                                                    self.no_whitespaces, 
                                                    self.whitespace_IDs)
        else:
            max_no_whitespaces = int(xp.amax(self.no_whitespaces))
            if self.swapaxes:
                y = xp.zeros((N, max_no_whitespaces, D), dtype=np.float32)
            else:
                y = xp.zeros((max_no_whitespaces, N, D), dtype=np.float32)
            y = _forward_gpu(IDs, x, y, 
                            self.no_whitespaces, 
                            self.whitespace_IDs,
                            self.swapaxes)
        return y,
        
    
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, IDs = inputs
        gy = grad_outputs[0]
        
        gx   = xp.zeros_like(x)
        
        if xp is np:
            if self.swapaxes:
                gx = extract_words_cpu_funcs.backward_axes_swapped(IDs, gy, gx, 
                                                        self.no_whitespaces, 
                                                        self.whitespace_IDs)
            else:
                gx = extract_words_cpu_funcs.backward(IDs, gy, gx, 
                                                    self.no_whitespaces, 
                                                    self.whitespace_IDs)
        else:
            gx = _backward_gpu(IDs, gy, gx, 
                               self.no_whitespaces, 
                               self.whitespace_IDs,
                               self.swapaxes)
            
        return gx, None
        
def extractWords(x, IDs, no_whitespaces, whitespace_IDs, swapaxes=False):
    return ExtractWords(no_whitespaces, whitespace_IDs, swapaxes)(x, IDs)


def getNoWhitespaces(IDs, whitespace_IDs):
    xp = cuda.get_array_module(IDs, whitespace_IDs)
    if xp is np:
        return cpu.utils.getNoWhitespaces(IDs, whitespace_IDs)
    else:
        return gpu.utils.getNoWhitespaces(IDs, whitespace_IDs)