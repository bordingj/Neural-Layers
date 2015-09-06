
import numpy as np
cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel import prange
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lstm_apply_nonlinearity(np.float32_t[:,:] z, int out_size):
    
    cdef:
        int N = z.shape[0]
        int M = z.shape[1]
        np.intp_t cut = out_size*3
        np.intp_t i, j
        
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if j < cut:
                z[i,j] = 1/(1+math.exp(-z[i,j]))
            else:
                z[i,j] = math.tanh(z[i,j])
    return np.array(z, np.float32, copy=False, order='C')

            
@cython.boundscheck(False)
@cython.wraparound(False)
def lstm_final_mem_cell(np.float32_t[:,:] z, np.float32_t[:,:] c_tm1, 
                        np.float32_t[:,:] c, np.float32_t[:,:] h):
    cdef:
        int N = c.shape[0]
        int M = c.shape[1]
        np.intp_t i, j
        np.float32_t i_t, f_t, o_t, c_tilde_t, c_k
        
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            i_t = z[i,j]
            f_t = z[i,(j+M)]
            o_t = z[i,(j+M*2)]
            c_tilde_t = z[i,(j+M*3)]
            
            c_k = f_t*c_tm1[i,j] + i_t*c_tilde_t
            
            c[i,j] = c_k

            h[i,j] = math.tanh(c_k)*o_t

@cython.boundscheck(False)
@cython.wraparound(False)
def lstm_backward_finalmem_and_nonlinearities(np.float32_t[:,:] c, 
                                           np.float32_t[:,:] z, 
                                           np.float32_t[:,:] gh, 
                                           np.float32_t[:,:] gc, 
                                           np.float32_t[:,:] c_tm1, 
                                           bint gh_is_none, bint gc_is_none):
    cdef:
        int N = c.shape[0]
        int M = c.shape[1]
        np.intp_t i, j
        np.float32_t i_t, f_t, o_t, c_tilde_t, tanh_c_k, gh_k, gc_k, gc_tm1_k, gi_k, gf_k
        
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            i_t = z[i,j]
            f_t = z[i,(j+M)]
            o_t = z[i,(j+M*2)]
            c_tilde_t = z[i,(j+M*3)]
            
            tanh_c_k = math.tanh(c[i,j])
            
            if gh_is_none:
                gh_k = 0.0
            else:
                gh_k = gh[i,j]
            if gc_is_none:
                gc_k = 0.0
            else:
                gc_k = gc[i,j]

            gc_tm1_k = gh_k * o_t * (1 - tanh_c_k**2)+gc_k
            c[i,j] = gc_tm1_k*f_t #we use the memory for c as gc_tm1
            
            gi_k = gc_tm1_k* c_tilde_t * i_t * (1-i_t);

            z[i,(j+M*3)] = gc_tm1_k* i_t * (1-c_tilde_t*c_tilde_t);

            gf_k = gc_tm1_k* c_tm1[i,j] * f_t * (1-f_t);

            z[i,(j+M*2)] = gh_k* tanh_c_k * o_t * (1-o_t);

            z[i,j] = gi_k;

            z[i,(j+M)] = gf_k;