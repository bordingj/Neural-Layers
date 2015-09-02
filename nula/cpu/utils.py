
import numpy as np
from numba import jit
import math

def weight_initialization(in_size, out_size, scale):
    return np.random.normal(0, scale * np.sqrt(1. / in_size),
            (out_size, in_size)).astype(np.float32)



@jit('f4[:,:](f4[:,:], f4[:,:])')
def relu(x, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = 1e-6
    return np.array(out, np.float32, copy=False, order='C')


@jit('f4[:,:](f4[:,:], f4[:,:], f4[:,:])')
def grelu(x, gy, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = 1e-6
    return np.array(out, np.float32, copy=False, order='C')

@jit('f4[:,:](f4[:,:], f4[:,:], f4)')
def leakyrelu(x, out, alpha=0.1):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = x[i,j]*alpha
    return np.array(out, np.float32, copy=False, order='C')

@jit('f4[:,:](f4[:,:], f4[:,:],f4[:,:], f4)')
def gleakyrelu(x, gy, out, alpha=0.1):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = gy[i,j]*alpha
    return np.array(out, np.float32, copy=False, order='C')

@jit('f4[:,:](f4[:,:], f4[:,:])')
def tanh(x, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = math.tanh(x[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@jit('f4[:,:](f4[:,:], f4[:,:], f4[:,:])')
def gtanh(gy, y, out):
    N, M = y.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = gy[i,j] * (1 - y[i,j] * y[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@jit('f4[:,:](f4[:,:], f4[:,:])')
def sigmoid(x, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = 1/(1+math.exp(-x[i,j]))
    return np.array(out, np.float32, copy=False, order='C')

@jit('f4[:,:](f4[:,:], f4[:,:], f4[:,:])')
def gsigmoid(gy, y, out):
    N, M = y.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = gy[i,j] * y[i,j] * (1 - y[i,j])
    return np.array(out, np.float32, copy=False, order='C')


@jit('f4[:,:](f4[:,:], i4[:,:], f4[:,:], b1)')
def hotdot(a, indices, out, 
           dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: x.dot(a.T), where x is a K-hot encoded matrix 
    
    """
    
    H, D = a.shape
    N, K = indices.shape
    
    if dont_add:
        for i in range(N):
            for j in range(H):
                out[i,j] = 0
        
    if K > 1:
        for i in range(N):
            for k in range(K):
                idx = indices[i,k]
                for j in range(H):
                    out[i,j] += a[j,idx]
    else:
        for i in range(N):
            idx = indices[i,1]
            for j in range(H):
                out[i,j] += a[j,idx]     
    return out
        
        
@jit('f4[:,:](f4[:,:], i4[:,:], f4[:,:], b1)')
def dothot(a, indices, 
           out, dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: a.T.dot(x), where x is a K-hot encoded matrix 
    
    """
    N, H = a.shape
    _N, K = indices.shape
    
    if _N != N:
        raise ValueError( 'a.shape[0] != idx.shape[0]' )

    if dont_add:
        M = out.shape[1]
        for i in range(H):
            for j in range(M):
                out[i,j] = 0
        
    if K > 1:
        for j in range(N):
            for k in range(K):
                idx = indices[j,k]
                for i in range(H):
                    out[i,idx] += a[j,i]
    else:
        for j in range(N):
            idx = indices[j,1]
            for i in range(H):
                out[i,idx] += a[j,i]
    return out
