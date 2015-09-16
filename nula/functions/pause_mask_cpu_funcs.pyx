
import numpy as np
cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel import prange
            
@cython.boundscheck(False)
@cython.wraparound(False)
def forward(np.float32_t[:,:] x, np.int32_t[:] indices, np.int32_t[:] pause_indices, 
            np.float32_t[:,:] row_masks, np.float32_t[:,:] y):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        int no_pause_indices = pause_indices.shape[0]
        np.intp_t i, k, j
        bint was_a_pause
        
    for i in prange(N, schedule='guided', nogil=True):
        for k in range(no_pause_indices):
            was_a_pause = 0
            if indices[i] == pause_indices[k]:
                was_a_pause = 1
                break
        if was_a_pause:
            row_masks[i] = 1.0
            for j in range(M):
                y[i,j] = x[i,j]
        else:
            row_masks[i] = 0.0
            for j in range(M):
                y[i,j] = 0.0
                
    return np.array(y, np.float32, copy=False, order='C')