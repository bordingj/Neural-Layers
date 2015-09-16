import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def crossentropy(np.float32_t[:,:] probs, np.int32_t[:] t):
    cdef:
        int N = probs.shape[0]
        np.float32_t logsum_correct_probs = 0
        np.intp_t i, k
    
    for i in prange(N, schedule='guided', nogil=True):
        k = <np.intp_t>t[i]
        logsum_correct_probs += log(min(max(probs[i,k], 1e-8), 1.0))
        
    return -np.float32(1.0/N)*logsum_correct_probs
        
        
@cython.boundscheck(False)
@cython.wraparound(False)
def gsoftmaxCrossentropy(np.float32_t[:,:] y, np.int32_t[:] t, np.float32_t coef):
    cdef:
        int N = y.shape[0]
        int M = y.shape[1]
        np.intp_t i, k, j
        
    for i in prange(N, schedule='guided', nogil=True):
        k = <np.intp_t>t[i]
        y[i,k] -= 1.0
        for j in range(M):
            y[i,j] *= coef 
    return np.array(y, np.float32, copy=False, order='C')