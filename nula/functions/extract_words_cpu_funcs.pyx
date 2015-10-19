
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def forward(np.int32_t[:,:,:] IDs, 
                np.float32_t[:,:,:] x, np.float32_t[:,:,:] y,
                np.int32_t[:] no_whitespaces, np.int32_t[:] whitespace_IDs):
    cdef:
        int no_whitespace_ids = whitespace_IDs.shape[0]
        int T = x.shape[0]
        int N = x.shape[1]
        int M = x.shape[2]
        np.intp_t t, i, j, k, no_whitespaces_i, current_max_whitespaces
    
    for j in range(M):
        for i in range(N):
            no_whitespaces_i = 0
            current_max_whitespaces = no_whitespaces[i];
            for t in range(T):
                if no_whitespaces_i < current_max_whitespaces:
                    for k in range(no_whitespace_ids):
                        if IDs[t,i,0] == whitespace_IDs[k]:
                            y[no_whitespaces_i,i,j] = x[t,i,j]
                            no_whitespaces_i += 1
                            break
                else:
                    break
    
    return np.array(y, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def forward_axes_swapped(np.int32_t[:,:,:] IDs, 
                np.float32_t[:,:,:] x, np.float32_t[:,:,:] y,
                np.int32_t[:] no_whitespaces, np.int32_t[:] whitespace_IDs):
    cdef:
        int no_whitespace_ids = whitespace_IDs.shape[0]
        int T = x.shape[0]
        int N = x.shape[1]
        int M = x.shape[2]
        np.intp_t t, i, j, k, no_whitespaces_i, current_max_whitespaces
    
    for j in range(M):
        for i in range(N):
            no_whitespaces_i = 0
            current_max_whitespaces = no_whitespaces[i];
            for t in range(T):
                if no_whitespaces_i < current_max_whitespaces:
                    for k in range(no_whitespace_ids):
                        if IDs[t,i,0] == whitespace_IDs[k]:
                            y[i,no_whitespaces_i,j] = x[t,i,j]
                            no_whitespaces_i += 1
                            break
                else:
                    break
    
    return np.array(y, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def backward(np.int32_t[:,:,:] IDs, 
                np.float32_t[:,:,:] gy, np.float32_t[:,:,:] gx,
                np.int32_t[:] no_whitespaces, np.int32_t[:] whitespace_IDs):
    cdef:
        int no_whitespace_ids = whitespace_IDs.shape[0]
        int T = gx.shape[0]
        int N = gx.shape[1]
        int M = gx.shape[2]
        np.intp_t t, i, j, k, no_whitespaces_i, current_max_whitespaces
    
    for j in range(M):
        for i in range(N):
            no_whitespaces_i = 0
            current_max_whitespaces = no_whitespaces[i];
            for t in range(T):
                if no_whitespaces_i < current_max_whitespaces:
                    for k in range(no_whitespace_ids):
                        if IDs[t,i,0] == whitespace_IDs[k]:
                            gx[t,i,j] = gy[no_whitespaces_i,i,j]
                            no_whitespaces_i += 1
                            break
                else:
                    break
    
    return np.array(gx, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def backward_axes_swapped(np.int32_t[:,:,:] IDs, 
                np.float32_t[:,:,:] gy, np.float32_t[:,:,:] gx,
                np.int32_t[:] no_whitespaces, np.int32_t[:] whitespace_IDs):
    cdef:
        int no_whitespace_ids = whitespace_IDs.shape[0]
        int T = gx.shape[0]
        int N = gx.shape[1]
        int M = gx.shape[2]
        np.intp_t t, i, j, k, no_whitespaces_i, current_max_whitespaces
    
    for j in range(M):
        for i in range(N):
            no_whitespaces_i = 0
            current_max_whitespaces = no_whitespaces[i];
            for t in range(T):
                if no_whitespaces_i < current_max_whitespaces:
                    for k in range(no_whitespace_ids):
                        if IDs[t,i,0] == whitespace_IDs[k]:
                            gx[t,i,j] = gy[i,no_whitespaces_i,j]
                            no_whitespaces_i += 1
                            break
                else:
                    break
    
    return np.array(gx, np.float32, copy=False, order='C')