
import numpy as np
cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel import prange
            
@cython.boundscheck(False)
@cython.wraparound(False)
def get_ga(np.float32_t[:,:] gy, np.float32_t[:,:] x, np.float32_t[:] ga):
    cdef:
        int N = gy.shape[0]
        int M = gy.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        ga[i] = 0
        for j in range(M):
            ga[i] += gy[i,j]*x[i,j]
    return np.array(ga, np.float32, copy=False, order='C')