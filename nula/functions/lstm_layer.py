
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from nula import cpu
import chainer.functions as F

if cuda.available:
    import cupy as cp
    from nula import gpu

        
class LSTMLayer(function.Function):
    """
    This is a parametric LSTM layer as described in http://arxiv.org/pdf/1410.4615v3.pdf
    """
    def __init__(self, in_size, out_size,
                 Wscale=1.0, Vscale=1.0, 
                 nobias=False, bias=0.0, forget_bias=1.0):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
        self.forget_bias = np.float32(forget_bias)
            
        #initialize weight matrices 
        self.W = cpu.utils.weight_initialization(in_size, out_size*4, Wscale)
        self.gW = np.empty_like(self.W)
        
        self.V = cpu.utils.weight_initialization(out_size, out_size*4, Vscale)
        self.gV = np.empty_like(self.V)
        
        if not self.nobias:
            self.b = np.empty((1, out_size*4), dtype=np.float32)
            self.b.fill(self.bias)
            self.b[0,out_size*2:out_size*3] = self.forget_bias
            self.gb = np.empty_like(self.b)
    
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
        
        z       = xp.empty((batchsize,self.out_size*4),dtype=np.dtype('float32'))

        if xp is np:
            z  = x.dot(self.W.T, out=z)
            z += h_tm1.dot(self.V.T)
            if not self.nobias:
                z += self.b
        else:
            z = cp.dot(x, self.W.T, out=z)
            gpu.utils.dot_add(A=h_tm1, B=self.V, C=z, transb=True)
            if not self.nobias:
                gpu.utils.addVec2Mat(z, self.b)
        
        self.lstm_fun = F.LSTM()
        c, h = self.lstm_fun.forward(inputs=(c_tm1, z))
        self.z = z
        return h, c

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gh, gc = grad_outputs
        x, h_tm1, c_tm1 = inputs
        
        gc_tm1, gz = self.lstm_fun.backward(inputs=(c_tm1, self.z), 
                                     grad_outputs=(gc, gh))
        
        batchsize = x.shape[0]        
        gh_tm1 = xp.empty_like(h_tm1)
        gx      = xp.empty((batchsize,self.in_size),dtype=np.dtype('float32'))
        
        if xp is np:
            gh_tm1 = np.dot(gz, self.V, out=gh_tm1)
            # compute gradient with respect to the input x
            gx = np.dot(gz, self.W, out=gx)
             # compute gradients of weight matrices
            self.gW += gz.T.dot(x)
            self.gV += gz.T.dot(h_tm1)
            if not self.nobias:
                gb_ones = xp.ones((1,batchsize), dtype=np.dtype('float32'))
                self.gb += np.dot(gb_ones, gz)
        else:
            gh_tm1 = cp.dot(gz, self.V, out=gh_tm1)
            # compute gradient with respect to the input x
            gx = cp.dot(gz, self.W, out=gx)
            # compute gradients of weight matrices
            gpu.utils.dot_add(gz, x, C=self.gW, transa=True)
            gpu.utils.dot_add(gz, h_tm1, C=self.gV, transa=True)
            if not self.nobias:
                gb_ones = xp.ones((1,batchsize), dtype=np.dtype('float32'))
                gpu.utils.dot_add(gb_ones, gz, C=self.gb)

        return gx, gh_tm1, gc_tm1