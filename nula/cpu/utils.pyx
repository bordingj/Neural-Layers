
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
def getNoWhitespaces(np.int32_t[:,:,:] IDs, np.int32_t[:] whitespace_IDs):
    cdef: 
        int no_whitespace_ids = whitespace_IDs.shape[0]
        int T = IDs.shape[0]
        int N = IDs.shape[1]
        np.int32_t[:] no_whitespaces = np.zeros((N,), dtype=np.int32)
        np.intp_t k, i, t
    
    for t in range(T):
        for i in prange(N, schedule='guided', nogil=True):
            for k in range(no_whitespace_ids):
                if IDs[t,i,0] == whitespace_IDs[k]:
                    no_whitespaces[i] += 1
    return np.array(no_whitespaces, np.int32, copy=False, order='C')
    
@cython.boundscheck(False)
@cython.wraparound(False)
def getBatchPerformance(np.float32_t[:,:] probs, np.int32_t[:] t):
    pred = np.fliplr(np.argsort(probs, axis=1).astype(np.int32))
    assert pred.shape[0] == t.shape[0]
    Ranks = np.empty((pred.shape[0],), dtype=np.int32)
    acc   = np.zeros((pred.shape[0],), dtype=np.float32)
    MRRs = np.empty((pred.shape[0],), dtype=np.float32)
    Recallat5 = np.zeros((pred.shape[0],), dtype=np.float32)
    Recallat10 = np.zeros((pred.shape[0],), dtype=np.float32)
    Recallat20 = np.zeros((pred.shape[0],), dtype=np.float32)
    
    cdef:
        int N = pred.shape[0]
        int M = pred.shape[1]
        np.intp_t i, j
        np.int32_t true_idx, rank
        np.int32_t[:,:] pred_view = pred
        np.int32_t[:] Ranks_view = Ranks
        np.float32_t[:] acc_view = acc
        np.float32_t[:] MRRs_view = MRRs
        np.float32_t[:] Recallat5_view = Recallat5
        np.float32_t[:] Recallat10_view = Recallat10
        np.float32_t[:] Recallat20_view = Recallat20
                 
        
    for i in prange(N, schedule='guided', nogil=True):
        true_idx = t[i]
        for j in range(M):
            if pred_view[i,j]==true_idx:
                rank = j
                Ranks_view[i] = rank
                MRRs_view[i] = 1.0/(<np.float32_t>rank+1.0)
                if rank==0:
                    acc_view[i] = 1
                    Recallat5_view[i]  = 1
                    Recallat10_view[i] = 1
                    Recallat20_view[i] = 1
                elif rank < 5:
                    Recallat5_view[i]  = 1
                    Recallat10_view[i] = 1
                    Recallat20_view[i] = 1
                elif rank < 10:
                    Recallat10_view[i] = 1
                    Recallat20_view[i] = 1
                elif rank < 20:
                    Recallat20_view[i] = 1
                break
            
    return Ranks, acc.mean(), MRRs.mean(), Recallat5.mean(), Recallat10.mean(), Recallat20.mean()

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
