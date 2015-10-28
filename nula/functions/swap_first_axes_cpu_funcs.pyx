
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def forward(np.float32_t[:,:,:] x):
    cdef:
        int T = x.shape[0]
        int N = x.shape[1]
        int M = x.shape[2]
        np.float32_t[:,:,:] y = np.empty((N, T, M), dtype=np.float32)
        np.intp_t j, t, i
    
    for j in prange(M, schedule='guided', nogil=True):
        for t in range(T):
            for i in range(N):
                y[i,t,j] = x[t,i,j]
    return np.array(y, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def backward(np.float32_t[:,:,:] gy):
    cdef:
        int N = gy.shape[0]
        int T = gy.shape[1]
        int M = gy.shape[2]
        np.float32_t[:,:,:] gx = np.empty((T, N, M), dtype=np.float32)
        np.intp_t j, t, i

    for j in prange(M, schedule='guided', nogil=True):
        for t in range(T):
            for i in range(N):
                gx[t,i,j] = gy[i,t,j]
    return np.array(gx, np.float32, copy=False, order='C')