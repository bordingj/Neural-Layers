
import numpy as np
import math

from chainer import cuda
import cupy as cp


@cp.util.memoize(for_each_device=True)
def _GetNoWhitespaces_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void GetNoWhitespaces(const int* IDs, const int* whitespace_IDs, int* no_whitespaces,
                     const int no_whitespace_ids, const int T, const int N){
            
        __shared__ int shared_vec[16];
            
        if (blockIdx.y < N){
            int no_whitespaces_i = 0;
            int n; 
            for (int t = threadIdx.x; t < T; t += 16){
                n = t*N+blockIdx.y;
                for (int k=0; k<no_whitespace_ids; k++){
                    if (IDs[n] == whitespace_IDs[k]){
                        no_whitespaces_i += 1;         
                        break;
                    }               
                } 
            }
            shared_vec[threadIdx.x] = no_whitespaces_i;
            __syncthreads();
            
            if (threadIdx.x == 0){
                for (int j=0; j<16; j++){
                    no_whitespaces[blockIdx.y] += shared_vec[j];
                }
            }
        }
    }
    """)
    return kernel_code.get_function('GetNoWhitespaces')

def getNoWhitespaces(IDs, whitespace_IDs):
    no_whitespace_ids = whitespace_IDs.shape[0]
    T = IDs.shape[0]
    N = IDs.shape[1]
    no_whitespaces = cp.zeros((N,), dtype=np.int32)
    
    _GetNoWhitespaces = _GetNoWhitespaces_kernel()
    
    bdim, gdim = (16,1,1), (1,N,1)

    _GetNoWhitespaces(grid=gdim, block=bdim,
          args=(IDs, whitespace_IDs, no_whitespaces,
                no_whitespace_ids, T, N)
            )
    return no_whitespaces


@cp.util.memoize(for_each_device=True)
def _get_Relu_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void ReLU(float* x, float *y, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
                 
        if (i < N){
            y[i] = fmaxf(0.000001f, x[i]);
        }
    }
    """)
    return kernel_code.get_function('ReLU')

def relu(x, out):
    """
    This kernel is the rectifier unit max(x, 1e-6)
    """
    
    _ReLU = _get_Relu_kernel()
    
    bdim, gdim = Get_bdim_and_gdim1D(x.size)   
    
    _ReLU(grid=gdim, block=bdim,
          args=(x, out,
                np.int32(x.size))
            )
    return out
    
@cp.util.memoize(for_each_device=True)
def _get_gRelu_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void gReLU(float* x, float *gy, float *out, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
     
        if (i < N){
            out[i] = (x[i] >= 0.000001f) ? gy[i] : 0.000001f*gy[i];
        }
    }
    """)
    return kernel_code.get_function('gReLU')
     
     
def grelu(x, gy, out):
    """
    In:
        gy: gradient of the output y
        x: input x
    This kernel is the hadamard product of gy and the derivative of the rectifier unit 
    with respect to its input x
    """
    _gReLU = _get_gRelu_kernel()
    
    bdim, gdim = Get_bdim_and_gdim1D(x.size)   
    
    _gReLU(grid=gdim, block=bdim,
           args=(x, gy, out,
                 np.int32(x.size))
            )
    return out
    
@cp.util.memoize(for_each_device=True)
def _get_LeakyReLU_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void LeakyReLU(float* x, float *y, float alpha, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
     
        if (i < N){
            y[i] = (x[i] >= 0.000001f) ? x[i] : alpha*x[i];
        }
    }
    """)
    return kernel_code.get_function('LeakyReLU')

def leakyrelu(x, out, alpha=0.1):
    """
    This kernel is the leaky rectifier unit
    """
    
    _LeakyReLU = _get_LeakyReLU_kernel()
    bdim, gdim = Get_bdim_and_gdim1D(x.size)   
    
    _LeakyReLU(grid=gdim, block=bdim,
               args=(x, out, np.float32(alpha),
                np.int32(x.size))
                )
    return out
    
@cp.util.memoize(for_each_device=True)
def _get_gLeakyReLU_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void gLeakyReLU(float* x, float *gy, float *out, float alpha, 
                    const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
                 
        if (i < N){
            out[i] = (x[i] >= 0.000001f) ? gy[i] : alpha*gy[i];
        }
    }
    """)
    return kernel_code.get_function('gLeakyReLU')

def gleakyrelu(x, gy, out, alpha=0.1):
    """
    In:
        gy: gradient of the output y
        x: input x
    This kernel is the hadamard product of gy and the derivative of the leaky rectifier unit 
    with respect to its input x
    """
    _gLeakyReLU = _get_gLeakyReLU_kernel()
    bdim, gdim = Get_bdim_and_gdim1D(x.size)   
    _gLeakyReLU(grid=gdim, block=bdim,
               args=(x, gy, out, np.float32(alpha),
                                    np.int32(x.size))
                )
    return out
    
@cp.util.memoize(for_each_device=True)
def _get_Sigmoid_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void Sigmoid(float *x, float *y, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
     
        if (i < N){
            y[i] = 1/(1+exp(-x[i]));
        }
    }
    """)
    return kernel_code.get_function('Sigmoid')

def sigmoid(x, out):
    """
    This kernel is the logistic sigmoid function 1/(1+exp(-x))
    """
    _Sigmoid = _get_Sigmoid_kernel()
    bdim, gdim = Get_bdim_and_gdim1D(x.size)
    _Sigmoid(grid=gdim, block=bdim,
             args=(x, out,
             np.int32(x.size))
             )
    return out

@cp.util.memoize(for_each_device=True)
def _get_gSigmoid_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void gSigmoid(float *gy, float *y, float *out, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
     
        if (i < N){
            out[i] = y[i]*(1-y[i])*gy[i];
        }
    }
    """)
    return kernel_code.get_function('gSigmoid')

def gsigmoid(gy, y, out):
    """
    In:
        gy: gradient of the output y
    This kernel is the hadamard product of gy and the derivative of the sigmoid function 
    with respect to its input. 
    """
    _gSigmoid = _get_gSigmoid_kernel()
    bdim, gdim = Get_bdim_and_gdim1D(gy.size)
    
    _gSigmoid(grid=gdim, block=bdim,
              args=(gy, y, out,
                    np.int32(gy.size))
                )
    return out
    

@cp.util.memoize(for_each_device=True)
def _get_tanh_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void Tanh(float *x, float *y, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
                 
        if (i < N){
            y[i] = tanhf(x[i]);
        }
    }
    """)
    return kernel_code.get_function('Tanh')

def tanh(x, out):
    """
    This kernel is the tanh function
    """
    _Tanh = _get_tanh_kernel()
    bdim, gdim = Get_bdim_and_gdim1D(x.size)   
    _Tanh(grid=gdim, block=bdim,
          args=(x, out,
                np.int32(x.size))
                )
    return out
    
@cp.util.memoize(for_each_device=True)
def _get_gtanh_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void gTanh(float *gy, float *y, float *out, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
                 
        if (i < N){
            out[i] = (1-y[i]*y[i])*gy[i];
        }
    }
    """)
    return kernel_code.get_function('gTanh')

def gtanh(gy, y, out):
    """
    In:
        gy: gradient of the output y
    This kernel is the hadamard product of gy and the derivative of the tanh function 
    with respect to its input. 
    """
    
    _gTanh = _get_gtanh_kernel()
    bdim, gdim = Get_bdim_and_gdim1D(gy.size)
    
    _gTanh(grid=gdim, block=bdim,
           args=(gy, y, out,
                 np.int32(gy.size))
            )
    return out

@cp.util.memoize(for_each_device=True)
def _get_HotDot_kernels():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void HotDot1(float* a, float* out, int* indices, 
                         int K, int N, int H, int D, int B)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
                
        if (i < N && j < H){
            int n = i*H+j;
            if(B){
                out[n] = 0;
            }
            for (int k=0;k<K;k++){
                int idx = indices[i*K+k];
                out[n] += a[j*D+idx];
            }    
        }
    }
    
    extern "C" __global__
    void HotDot2(float* a, float* out, int* indices, 
                         int N, int H, int D, int B)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
    
        if (i < N && j < H){
            int n = i*H+j;
            int idx = indices[i];
            if (B){
                out[n] = a[j*D+idx];
            }else{
                out[n] += a[j*D+idx];
            }
        }
    }
    """)
    return kernel_code.get_function('HotDot1'), kernel_code.get_function('HotDot2')

def hotdot(a, indices, out=None, dont_add=False):
    """
    In:
        a: a pycuda gpuarray
        indices: hot indices a K-hot encoded matrix
    out:
        out: x.dot(a.T), where x is a K-hot encoded matrix 
    
    """
    HotDot1, HotDot2 = _get_HotDot_kernels()
    H, D = a.shape
    N, K = indices.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(H)
    elif H >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,H)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,H)
    if dont_add:
        B = np.int32(1)
    else:
        B = np.int32(0)
        
    if out is None:
        out = cp.empty((N,H), dtype=np.float32)
        B = np.int32(1)
    
    if K > 1:
        HotDot1(grid=gdim, block=bdim,
                 args=(a, out, indices,
                np.int32(K), np.int32(N), np.int32(H), np.int32(D), np.int32(B))
                )
    else:
        HotDot2(grid=gdim, block=bdim,
                 args=(a, out, indices,
                        np.int32(N), np.int32(H), np.int32(D), np.int32(B))
                )
        return out

@cp.util.memoize(for_each_device=True)
def _get_DotHot_kernels():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void DotHot1(float* a, float* out, int* indices, 
                         int K, int N, int H, int D, int B)
    {    
        int j = threadIdx.x + blockIdx.x * blockDim.x;
                
        if (j < H){    
            for (int i=0;i<N;i++){
                for (int k=0;k<K;k++){
                    int idx = indices[i*K+k];
                    out[j*D+idx] += a[i*H+j];
                }
            }
        }
    }
    
    extern "C" __global__
    void DotHot2(float* a, float* out, int* indices, 
                             int N, int H, int D, int B)
    {   
        int j = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (j < H){                
            for (int i=0;i<N;i++){
                int idx = indices[i];
                out[j*D+idx] += a[i*H+j];
            }
        }
    }
    """)
    return kernel_code.get_function('DotHot1'), kernel_code.get_function('DotHot2')



def dothot(a, indices, in_size, out=None, dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: a.T.dot(x), where x is a K-hot encoded matrix 
    
    """
    
    DotHot1, DotHot2 = _get_DotHot_kernels()
    N, H = a.shape
    _N, K = indices.shape
    if _N != N:
        raise ValueError( 'a.shape[0] != idx.shape[0]' )
        
    bdim, gdim = Get_bdim_and_gdim1D(H)
    if dont_add:
        B = np.int32(1)
    else:
        B = np.int32(0)
    
    if out is None:
        out = cp.zeros((H,in_size), dtype=np.float32)
    
    if K > 1:
        DotHot1(grid=gdim, block=bdim,
                 args=(a, out, indices,
            np.int32(K), np.int32(N), np.int32(H), np.int32(in_size), np.int32(B))
            )
    else:
        DotHot2(grid=gdim, block=bdim,
                 args=(a, out, indices,
                    np.int32(N), np.int32(H), np.int32(in_size), np.int32(B))
                )
    return out

@cp.util.memoize(for_each_device=True)
def _get_AddVec2Mat_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void AddVec2Mat(float* h, float* b, const int N, const int M)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
    
        if (i < N && j < M){
            h[i*M+j] += b[j];
        }
    }
    """)
    return kernel_code.get_function('AddVec2Mat')

def addVec2Mat(h, b):
    """
    In:
        h: a gpuarray matrix as shape NxH
        b: a gpuarray vector of shape 1xH (or Hx1)
    
    This kernel adds vector b to every row of matrix a
    """
    _AddVec2mat = _get_AddVec2Mat_kernel()
    N, M = h.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M) 

    _AddVec2mat(grid=gdim, block=bdim,
                   args=(h, b, np.int32(N), np.int32(M))
                   )
    return h

@cp.util.memoize(for_each_device=True)
def _get_Hadamard_kernel():
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void Hadamard(float *A, float *B, float *out, float scalar, const int N)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
                 
        if (i < N){
            out[i] = scalar*A[i]*B[i];
        }
    }
    """)
    return kernel_code.get_function('Hadamard')

def hadamard(A, B, out, scalar=np.float32(1.0)):
    """
    This kernel computes the elementwise product (hadamard product) of A and B scaled by scalar
    """
    bdim, gdim = Get_bdim_and_gdim1D(A.size)
    
    _Hadamard = _get_Hadamard_kernel()
    _Hadamard(grid=gdim, block=bdim,
                  args=(A, B, out, 
                        np.float32(scalar),
                        np.int32(A.size))
                    )
    return out

def Get_bdim_and_gdim1D(N):
    """
    returns 1d block and 1d grid dimensions for pycuda sourcemodule kernel call 
    """
    k = math.ceil(N/32.0)
    blocksize = min(512,k*32)
    bdim = (blocksize, 1, 1)
    gdim = (math.ceil(N/blocksize),1,1)
    return bdim, gdim


def Get_bdim_and_gdimRowVec(M):
    """
    returns 1d block and 1d grid dimensions for pycuda sourcemodule kernel call 
    """
    k = math.ceil(M/32)
    blocksize = min(512,k*32)
    bdim = (1, blocksize, 1)
    gdim = (1,math.ceil(M/blocksize),1)
    return bdim, gdim

def Get_bdim_and_gdimSmallNBigM(N, M):
    """
    returns 2d block and 2d grid dimensions for pycuda sourcemodule kernel call 
    """
    bdim = (8, 64, 1)
    gdim = (math.ceil(N/8.0),math.ceil(M/64.0),1)
    return bdim, gdim

def Get_bdim_and_gdim2D(N, M):
    """
    returns 2d block and 2d grid dimensions for pycuda sourcemodule kernel call 
    """
    bdim = (16, 32, 1)
    gdim = (math.ceil(N/16.0), math.ceil(M/32.0),1)
    return bdim, gdim

def dot_add(A, B, C, transa=False, transb=False, 
            alpha=1, beta=1):
    """
    This is just the blas-routine Sgemm:
    C = alpha*A.dot(B)+beta*C,
    where default alpha is 1 and default beta is 0
    """
    if transa:
        transa = cp.cuda.cublas.CUBLAS_OP_T
        l, n = A.shape
    else:
        transa = cp.cuda.cublas.CUBLAS_OP_N
        n, l = A.shape
    if transb:
        transb = cp.cuda.cublas.CUBLAS_OP_T
        m, k = B.shape
    else:
        transb = cp.cuda.cublas.CUBLAS_OP_N
        k, m = B.shape
    if l != k:
        raise ValueError('objects are not aligned')
    if C.shape != (n, m) or C.dtype != A.dtype:
        raise ValueError('invalid value for c')
    cp.cuda.cublas.sgemm(cp.cuda.Device(cuda.get_device(A)).cublas_handle, 
                transb, transa, m, n, k, np.float32(alpha), B.data.ptr,
                np.int32(B.shape[1]), A.data.ptr, np.int32(A.shape[1]),np.float32(beta), 
                C.data.ptr, np.int32(C.shape[1]))
    return C