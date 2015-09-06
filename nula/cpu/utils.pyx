
import numpy as np
cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel import prange

def weight_initialization(in_size, out_size, scale):
    return np.random.normal(0, scale * np.sqrt(1. / in_size),
            (out_size, in_size)).astype(np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
def relu(np.float32_t[:,:] x, np.float32_t[:,:] out):
    cdef: 
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = 1e-6
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def grelu(np.float32_t[:,:] x, np.float32_t[:,:] gy, np.float32_t[:,:] out):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = 1e-6
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def leakyrelu(np.float32_t[:,:] x, np.float32_t[:,:] out, np.float32_t alpha=0.1):
    cdef: 
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = x[i,j]*alpha
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def gleakyrelu(np.float32_t[:,:] x, np.float32_t[:,:] gy, np.float32_t[:,:] out, 
                np.float32_t alpha=0.1):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = gy[i,j]*alpha
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def tanh(np.float32_t[:,:] x, np.float32_t[:,:] out):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = math.tanh(x[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def gtanh(np.float32_t[:,:] gy, np.float32_t[:,:] y, np.float32_t[:,:] out):
    cdef:
        int N = y.shape[0]
        int M = y.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = gy[i,j] * (1 - y[i,j] * y[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sigmoid(np.float32_t[:,:] x, np.float32_t[:,:] out):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = 1/(1+math.exp(-x[i,j]))
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def gsigmoid(np.float32_t[:,:] gy, np.float32_t[:,:] y, np.float32_t[:,:] out):
    cdef:
        int N = y.shape[0]
        int M = y.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = gy[i,j] * y[i,j] * (1 - y[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def gSoftmaxCrossEntropy(np.float32_t[:,:] gx, np.int32_t[:] t):
    N = gx.shape[0]
    cdef np.intp_t i, k
    for i in prange(N, schedule='guided', nogil=True):
        k = <np.intp_t>t[i]
        gx[i,k] -= 1
        
@cython.boundscheck(False)
@cython.wraparound(False)
def hotdot(np.float32_t[:,:] a, np.int32_t[:,:] indices, np.float32_t[:,:] out, 
           bint dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: x.dot(a.T), where x is a K-hot encoded matrix 
    
    """
    
    cdef:
        int H = a.shape[0]
        int D = a.shape[1]
        int N = indices.shape[0]
        int K = indices.shape[1]
        cdef np.intp_t i, j, k, idx
    
    if dont_add:
        for i in prange(N, schedule='guided', nogil=True):
            for j in range(H):
                out[i,j] = 0
        
    if K > 1:
        for i in prange(N, schedule='guided', nogil=True):
            for k in range(K):
                idx = <np.intp_t>indices[i,k]
                for j in range(H):
                    out[i,j] += a[j,idx]
    else:
        for i in prange(N, schedule='guided', nogil=True):
            idx = <np.intp_t>indices[i,1]
            for j in range(H):
                out[i,j] += a[j,idx]     
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def dothot(np.float32_t[:,:] a, np.int32_t[:,:] indices, 
           np.float32_t[:,:] out, bint dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: a.T.dot(x), where x is a K-hot encoded matrix 
    
    """
    cdef:
        int N = a.shape[0]
        int H = a.shape[1]
        int _N = indices.shape[0]
        int K = indices.shape[1]
        np.intp_t i, j, k, idx

    if _N != N:
        raise ValueError( 'a.shape[0] != idx.shape[0]' )


        
    if dont_add:
        M = out.shape[1]
        for i in prange(H, schedule='guided', nogil=True):
            for j in range(M):
                out[i,j] = 0
        
    if K > 1:
        for j in prange(N, schedule='guided', nogil=True):
            for k in range(K):
                idx = <np.intp_t>indices[j,k]
                for i in range(H):
                    out[i,idx] += a[j,i]
    else:
        for j in prange(N, schedule='guided', nogil=True):
            idx = <np.intp_t>indices[j,1]
            for i in range(H):
                out[i,idx] += a[j,i]
    return np.array(out, np.float32, copy=False, order='C')
