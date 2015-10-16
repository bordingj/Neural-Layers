
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def forward(np.float32_t[:,:,:] x, np.float32_t[:,:] y, 
            np.int32_t[:] indices, int offset):
    cdef:
        int N = y.shape[0]
        int M = y.shape[1]
        np.intp_t i, t, j
        
    for i in range(N):
        t = indices[i]+offset
        for j in range(M):
            y[i,j] = x[t,i,j]
            
    return np.array(y, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def backward(np.float32_t[:,:,:] gx, np.float32_t[:,:] gy, 
             np.int32_t[:] indices, int offset):
    cdef:
        int N = gy.shape[0]
        int M = gy.shape[1]
        np.intp_t i, t, j
        
    for i in range(N):
        t = indices[i]+offset
        for j in range(M):
            gx[t,i,j] = gy[i,j]
            
    return np.array(gx, np.float32, copy=False, order='C')