
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula.functions import lstm_cpu_funcs
from nula import cpu

if cuda.available:
    import cupy as cp
    from nula import gpu        
        
class LSTMLayer(function.Function):
    """
    This is a parametric LSTM layer as described in http://arxiv.org/pdf/1410.4615v3.pdf
    """
    def __init__(self, in_size, out_size,
                 Wscale=1.0, Vscale=1.0, 
                 nobias=False, bias=0.0,
                 dropout=False):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
            
        #initialize weight matrices 
        self.W = cpu.utils.weight_initialization(in_size, out_size*4, Wscale)
        self.gW = np.empty_like(self.W)
        
        self.V = cpu.utils.weight_initialization(out_size, out_size*4, Vscale)
        self.gV = np.empty_like(self.V)
        
        if not self.nobias:
            self.b = np.empty((1, out_size*4), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)
        
        self.z = None
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'W', 'V', 'b'
        else:
            return 'W', 'V'

    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gW', 'gV', 'gb'
        else:
            return 'gW', 'gV'
    
    def check_type_forward(self, in_types):
        x, h_tm1, c_tm1 = in_types
        
        type_check.expect(
            h_tm1.shape == c_tm1.shape,
            c_tm1.shape == (x.shape[0], self.out_size),
            x.shape[1] == self.in_size,

            c_tm1.dtype == np.float32,
            h_tm1.dtype == c_tm1.dtype,
        )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, h_tm1, c_tm1 = inputs
        
        batchsize = x.shape[0]
        
        self.z       = xp.empty((batchsize,self.out_size*4),dtype=np.dtype('float32'))
        self.c       = xp.empty((batchsize,self.out_size),dtype=np.dtype('float32'))
        self.h       = xp.empty((batchsize,self.out_size),dtype=np.dtype('float32'))

        if xp is np:
            self.z = np.dot(x, self.W.T, out=self.z)
            self.z += np.dot(h_tm1, self.V.T)
            if not self.nobias:
                self.z += self.b
            
            _lstm_forward_cpu(z=self.z, c_tm1=c_tm1, c=self.c, 
                         h=self.h, out_size=self.out_size)
        else:
            self.z = cp.dot(x, self.W.T, out=self.z)
            gpu.utils.dot_add(A=h_tm1, B=self.V, C=self.z, transb=True)
            if not self.nobias:
                gpu.utils.addVec2Mat(self.z, self.b)
            _lstm_forward_gpu(z=self.z, c_tm1=c_tm1, c=self.c, 
                         h=self.h, out_size=self.out_size)
            
        return self.h, self.c

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gh, gc = grad_outputs
        x, h_tm1, c_tm1 = inputs
        
        if gh is None:
            gh = xp.array([[0]], dtype=np.float32)
            gh_is_none = 1
        else:
            gh_is_none = 0
        if gc is None:
            gc = xp.array([[0]], dtype=np.float32)
            gc_is_none = 1
        else:
            gc_is_none = 0
        
        gc_tm1 = self.c
        
        batchsize = x.shape[0]
        
        gx      = xp.empty((batchsize,self.in_size),dtype=np.dtype('float32'))
        
        if xp is np:
            _lstm_backward_cpu(c=self.c, z=self.z, gh=gh, 
                          gc=gc, c_tm1=c_tm1,
                          gc_is_none=gc_is_none, gh_is_none=gh_is_none)
            gz = self.z
            gh_tm1 = np.dot(gz, self.V, out=self.h)
            # compute gradient with respect to the input x
            gx = np.dot(gz, self.W, out=gx)
             # compute gradients of weight matrices
            self.gW += gz.T.dot(x)
            self.gV += gz.T.dot(h_tm1)
            if not self.nobias:
                gb_ones = xp.ones((1,batchsize), dtype=np.dtype('float32'))
                self.gb += np.dot(self.gb_ones, gz)
        else:
            _lstm_backward_gpu(c=self.c, z=self.z, gh=gh, 
                          gc=gc, c_tm1=c_tm1,
                          gc_is_none=gc_is_none, gh_is_none=gh_is_none)

            gz = self.z
            gh_tm1 = cp.dot(gz, self.V, out=self.h)
            # compute gradient with respect to the input x
            gx = cp.dot(gz, self.W, out=gx)
            # compute gradients of weight matrices
            gpu.utils.dot_add(gz, x, C=self.gW, transa=True)
            gpu.utils.dot_add(gz, h_tm1, C=self.gV, transa=True)
            if not self.nobias:
                gb_ones = xp.ones((1,batchsize), dtype=np.dtype('float32'))
                gpu.utils.dot_add(gb_ones, gz, C=self.gb)

        return gx, gh_tm1, gc_tm1




@cp.util.memoize(for_each_device=True)
def _get_lstm_forward_kernels():
    
    kernel_code = cp.carray.compile_with_cache("""
    extern "C" __global__
    void lstm_apply_nonlinearity(float *z, int out_size, const int N, const int M)
    {   
        int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
        int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                         
        if (idx_i < N && idx_j < M){
            int k = idx_i*M+idx_j;       
            z[k] = (idx_j < (out_size*3)) ? 1/(1+exp(-z[k])) : tanhf(z[k]);          
        }
    }
        
    extern "C" __global__
    void lstm_final_mem_cell(float *z, float *c_tm1, float *c, float* h, 
                                            const int N, const int M)
    {   
        int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
        int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                
        if (idx_i < N && idx_j < M){
            int k = idx_i*M+idx_j;
            int t = idx_i*M*4+idx_j;
                                
            float *i       = &z[0];
            float *f       = &z[M];
            float *o       = &z[M*2];
            float *c_tilde = &z[M*3];    
        
            float c_k = f[t] * c_tm1[k] + i[t] * c_tilde[t];
            c[k]      = c_k;
            h[k]      = tanhf( c_k  ) * o[t];
                                
        }
    }
    """)
    return kernel_code.get_function('lstm_apply_nonlinearity'), kernel_code.get_function('lstm_final_mem_cell')

def _lstm_forward_gpu(z, c_tm1, c, h, out_size):
    
    LSTM_apply_nonlinearity_func, LSTM_final_mem_cell_func = _get_lstm_forward_kernels()
    N, M = z.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)   
    
    LSTM_apply_nonlinearity_func(grid=gdim, block=bdim,
                           args=(z, np.int32(out_size),
                            np.int32(N), np.int32(M))
                            )
    N, M = h.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)
    
    LSTM_final_mem_cell_func(grid=gdim, block=bdim,
                        args=(z, c_tm1, c, h, 
                              np.int32(N), np.int32(M))
                              )

@cp.util.memoize(for_each_device=True)
def _get_lstm_backward_kernel():
    
    kernel_code = cp.carray.compile_with_cache("""
    extern "C"__global__
    void lstm_backward_finalmem_and_nonlinearities(
                            float *z,
                            float *gh,
                            float *c,
                            float *c_tm1,
                            float *gc,
                            int gc_is_none,
                            int gh_is_none,
                            const int N, const int M)
    {   
        int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
        int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                         
        if (idx_i < N && idx_j < M){
            int k = idx_i*M+idx_j;
            int t = idx_i*M*4+idx_j;
                            
            float *i       = &z[0];
            float *f       = &z[M];
            float *o       = &z[M*2];
            float *c_tilde = &z[M*3];
                            
            float gc_k = (gc_is_none) ? 0.0 : gc[k];
            float gh_k = (gh_is_none) ? 0.0 : gh[k];
                            
            float tanh_c_k = tanhf(c[k]);
            float gc_tm1_k = gh_k * o[t] * (1 - tanh_c_k*tanh_c_k) + gc_k;
            c[k]           = gc_tm1_k*f[t];
            float gi_k     = gc_tm1_k* c_tilde[t] * i[t] * (1-i[t]);
            c_tilde[t]     = gc_tm1_k* i[t] * (1-c_tilde[t]*c_tilde[t]);
            float gf_k     = gc_tm1_k* c_tm1[k] * f[t] * (1-f[t]);
            o[t]           = gh_k* tanh_c_k * o[t] * (1-o[t]);
            i[t]           = gi_k;
            f[t]           = gf_k;
        }
    }
    """)
    return kernel_code.get_function('lstm_backward_finalmem_and_nonlinearities')

def _lstm_backward_gpu(c, z, gh, gc, c_tm1, gc_is_none, gh_is_none):  
    
    LSTM_backward_finalmem_and_nonlinearities = _get_lstm_backward_kernel()
    N, M = c.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M)

    LSTM_backward_finalmem_and_nonlinearities(grid=gdim, block=bdim,
                     args=(z, gh, c, 
                     c_tm1, gc, np.int32(gc_is_none), np.int32(gh_is_none),
                     np.int32(N), np.int32(M))
                     )

def _lstm_forward_cpu(z, c_tm1, c, h, out_size):
    lstm_cpu_funcs.lstm_apply_nonlinearity(z=z, out_size=out_size)
    #final memory cell
    lstm_cpu_funcs.lstm_final_mem_cell(z=z, c_tm1=c_tm1, c=c, h=h)

def _lstm_backward_cpu(c, z, gh, gc, c_tm1, 
                       gh_is_none, gc_is_none):
    lstm_cpu_funcs.lstm_backward_finalmem_and_nonlinearities(c, z, gh, gc, c_tm1, 
                                           gh_is_none, gc_is_none)