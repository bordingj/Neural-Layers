

import numpy as np
from chainer import cuda
from nula import function
from chainer.utils import type_check

from nula import cpu

if cuda.available:
    import cupy as cp
    from nula import gpu


class SimpleLayer(function.Function):
    """
    This function is used to compute
    y = f(x.dot(W.T)+b)
    where x is an input matrices and W and b are parameters
    f is a non-linear elementwise activation function
     
    In:
        int in_size: number of columns in the input matrix;
        int out_size: number of columns in the output matrix (e.g. number of hidden states)
        str act_fun: activation function for the layer (default='tanh')
                    available activation functions: ('tanh', 'sigmoid', 'relu', 'leakyrelu')
        float Wscale: scale of the initialized weight matrix W (default=1.0)
        bool bias: if true the layer will have a bias parameter vector b (default=True)
        bool hot: if true we assumes that the input matrix x is K-hot encoded. Hence, the function should
                    be feed hot indices instead of a full matrix (default=False).
        Note that the function is non-differentiable with respect to its input if hot is True
    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0,
                 nobias=False,
                 bias=0.0,
                 hot=False):
    
        self.bias = bias
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
        self.hot = hot
        self.act_func_str = act_func.lower()
         
        #initialize W weight matrix
        self.W = cpu.utils.weight_initialization(in_size, out_size, Wscale)
        self.gW = np.empty_like(self.W)
                       
        if not self.nobias:
            self.b = np.empty((1, out_size), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)
 
        available_act_funcs = {
                    'sigmoid': (cpu.utils.sigmoid, cpu.utils.gsigmoid),
                    'tanh': (cpu.utils.tanh, cpu.utils.gtanh),
                    'relu': (cpu.utils.relu, cpu.utils.grelu),
                    'leakyrelu': (cpu.utils.leakyrelu, cpu.utils.gleakyrelu),
                    'linear': (None, None)
                    }
 
        available_cu_act_funcs = {
            'sigmoid': (gpu.utils.sigmoid, gpu.utils.gsigmoid),
            'tanh': (gpu.utils.tanh, gpu.utils.gtanh),
            'relu': (gpu.utils.relu, gpu.utils.grelu),
            'leakyrelu': (gpu.utils.leakyrelu, gpu.utils.gleakyrelu),
            'linear': (None, None)
        }
         
        self.act_func = available_act_funcs[self.act_func_str][0]
        self.gact_func = available_act_funcs[self.act_func_str][1]
 
        self.act_func_gpu = available_cu_act_funcs[self.act_func_str][0]
        self.gact_func_gpu = available_cu_act_funcs[self.act_func_str][1]
        
        self.z = None
        
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'W', 'b'
        else:
            return 'W',
 
    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gW', 'gb'
        else:
            return 'gW',

    def prepare(self, batchsize, on_gpu):
        xp = cp if on_gpu else np
        self.z  = xp.empty((batchsize,self.out_size), dtype=np.dtype('float32'))
        if not self.hot:
            self.gx = xp.empty((batchsize,self.in_size), dtype=np.dtype('float32'))
        if not self.nobias:
            self.gb_ones = cp.ones((1,batchsize),dtype=np.dtype('float32'))

    def check_type_forward(self, in_types):
        x, = in_types
        
        if not self.hot:
            type_check.expect(
                x.shape[1] == self.in_size,
                x.dtype == np.float32,
            )
        else:
            type_check.expect(
                x.shape[1] <= self.in_size,
                x.dtype == np.int32,
            )
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        N = x.shape[0]

        if self.z is None or self.z.shape[0] != N or type(self.z) != type(x):
            self.prepare(N, (xp == cp))
        
        if xp is np:
            self.act_func = self.act_func
            self.gact_func = self.gact_func
            #Linear function
            if self.hot: # if x is hot indices (k-hot-encoding)
                cpu.utils.hotdot(self.W, x, out=self.z, dont_add=True)
            else:
                z = np.dot(x, self.W.T, out=self.z)
            if not self.nobias:
                z += self.b
        else:
            self.act_func = self.act_func_gpu
            self.gact_func = self.gact_func_gpu
            if self.hot:
                gpu.utils.hotdot(self.W, x, out=self.z, dont_add=True)
            else:
                z = cp.dot(x, self.W.T, out=self.z)
            if not self.nobias:
                gpu.utils.addVec2Mat(self.z, self.b)
        
        #apply non-linear activation
        if self.act_func_str in ('tanh', 'sigmoid'):
            h = self.act_func(x=self.z, out=self.z)
            self.h = h #save h for backpropagation
        elif self.act_func_str in ('leakyrelu', 'relu'):
            h = xp.empty_like(self.z)
            h = self.act_func(x=self.z, out=h)
        elif self.act_func_str == 'linear':
            h = self.z
        else:
            raise NotImplementedError('the activation function is not available')
        return h,
 
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gh = grad_outputs[0]
        x = inputs[0]
        
        if self.act_func_str in ('tanh', 'sigmoid'):
            #backpropagate non-linearities
            gz = self.gact_func(gy=gh, y=self.h, out=self.h)
        elif self.act_func_str in ('leakyrelu', 'relu'):
            #backpropagate non-linearities
            gz = self.gact_func(x=self.z, gy=gh, out=self.z)
        elif self.act_func_str == 'linear':
            gz = gh
        else:
            raise NotImplementedError('the activation function is not available')

        #backpropagate linear function
        if xp is np:
            if self.hot:
                gx = None
                cpu.utils.dothot(gz, x, out=self.gW)
            else:
                gx = np.dot(gz, self.W, out=self.gx)
                self.gW += gz.T.dot(x)
            if not self.nobias:
                self.gb += np.dot(self.gb_ones, gz)
        else:
            if self.hot:
                gx = None
                gpu.utils.dothot(gz, x, in_size=self.in_size, out=self.gW)
            else:
                gx = cp.dot(gz, self.W, out=self.gx)
                gpu.utils.dot_add(A=gz, B=x, C=self.gW, transa=True)
            if not self.nobias:
                gpu.utils.dot_add(A=self.gb_ones, B=gz, C=self.gb)
         
        return gx, 
 
class SimpleLayer2Inputs(function.Function):
    """
    This function is used to compute
    y = f(x1.dot(W1.T)+x2.dot(W2.T)+b)
    where x1 and x2 are input matrices and W1, W2 and b are parameters
    x1 and x2 must have the same shapes
    f is a non-linear elementwise activation function
     
    In:
        int in_size: number of columns in the input matrix;
        int out_size: number of columns in the output matrix (e.g. number of hidden states)
        str act_fun: activation function for the layer (default='tanh')
                    available activation functions: ('tanh', 'sigmoid', 'relu', 'leakyrelu')
        float Wscale: scale of the initialized weight matrix W (default=1.0)
        bool bias: if true the layer will have a bias parameter vector b (default=True)
 
    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0,
                 nobias=False,
                 bias=0.0):
        
        if type(in_size) is int:
            self.in_size1 = in_size
            self.in_size2 = in_size
        else:
            assert len(in_size) == 2
            self.in_size1, self.in_size2 = in_size
        self.bias = bias
        self.nobias = nobias
        self.out_size = out_size
        self.act_func_str = act_func.lower()
         
        self.W1 = cpu.utils.weight_initialization(self.in_size1, out_size, Wscale)
        self.W2 = cpu.utils.weight_initialization(self.in_size2, out_size, Wscale)
        self.gW1 = np.empty_like(self.W1)
        self.gW2 = np.empty_like(self.W2)
                       
        if not self.nobias:
            self.b = np.empty((1, out_size), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)
 
        available_act_funcs = {
                    'sigmoid': (cpu.utils.sigmoid, cpu.utils.gsigmoid),
                    'tanh': (cpu.utils.tanh, cpu.utils.gtanh),
                    'relu': (cpu.utils.relu, cpu.utils.grelu),
                    'leakyrelu': (cpu.utils.leakyrelu, cpu.utils.gleakyrelu),
                    'linear': (None, None)
                    }
 
        available_cu_act_funcs = {
            'sigmoid': (gpu.utils.sigmoid, gpu.utils.gsigmoid),
            'tanh': (gpu.utils.tanh, gpu.utils.gtanh),
            'relu': (gpu.utils.relu, gpu.utils.grelu),
            'leakyrelu': (gpu.utils.leakyrelu, gpu.utils.gleakyrelu),
            'linear': (None, None)
        }
         
        self.act_func = available_act_funcs[self.act_func_str][0]
        self.gact_func = available_act_funcs[self.act_func_str][1]
 
        self.act_func_gpu  = available_cu_act_funcs[self.act_func_str][0]
        self.gact_func_gpu = available_cu_act_funcs[self.act_func_str][1]
        
        self.z = None
         
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'W1','W2', 'b'
        else:
            return 'W1','W2'
 
    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gW1', 'gW2','gb'
        else:
            return 'gW1', 'gW2'

    def prepare(self, batchsize, on_gpu):
        xp = cp if on_gpu else np
        self.z  = xp.empty((batchsize,self.out_size), dtype=np.dtype('float32'))
        self.gx1 = xp.empty((batchsize,self.in_size1), dtype=np.dtype('float32'))
        self.gx2 = xp.empty((batchsize,self.in_size2), dtype=np.dtype('float32'))
        if not self.nobias:
            self.gb_ones = cp.ones((1,batchsize),dtype=np.dtype('float32'))

    def check_type_forward(self, in_types):
        x1, x2 = in_types
        
        type_check.expect(
            x1.shape[1] == self.in_size1,
            x2.shape[1] == self.in_size2,
            x1.dtype == np.float32,
            x1.dtype == x2.dtype,
        )
        
        
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x1, x2 = inputs
        N = x1.shape[0]

        if self.z is None or self.z.shape[0] != N or type(self.z) != type(x1):
            self.prepare(N, (xp == cp))
        
        if xp is np:
            self.act_func = self.act_func
            self.gact_func = self.gact_func
            #Linear function
            z = np.dot(x1, self.W1.T, out=self.z)
            z += np.dot(x2, self.W2.T)
            if not self.nobias:
                z += self.b
        else:
            self.act_func = self.act_func_gpu
            self.gact_func = self.gact_func_gpu
            z = cp.dot(x1, self.W1.T, out=self.z)
            gpu.utils.dot_add(A=z,  B=self.W2, C=z, transb=True)
            if not self.nobias:
                gpu.utils.addVec2Mat(self.z, self.b)
        
        #apply non-linear activation
        if self.act_func_str in ('tanh', 'sigmoid'):
            h = self.act_func(x=z, out=self.z)
            self.h = h #save h for backpropagation
        elif self.act_func_str in ('leakyrelu', 'relu'):
            h = xp.empty_like(self.z)
            h = self.act_func(x=self.z, out=h)
        elif self.act_func_str == 'linear':
            h = self.z
        else:
            raise NotImplementedError('the activation function is not available')
        return h,
 
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gh = grad_outputs[0]
        x1, x2 = inputs
         
        if self.act_func_str in ('tanh', 'sigmoid'):
            #backpropagate non-linearities
            gz = self.gact_func(gy=gh, y=self.h, out=self.h)
        elif self.act_func_str in ('leakyrelu', 'relu'):
            #backpropagate non-linearities
            gz = self.gact_func(x=self.z, gy=gh, out=self.z)
        elif self.act_func_str == 'linear':
            gz = gh
        else:
            raise NotImplementedError('the activation function is not available')

        if xp is np:
            #backpropagate linear function
            gx1 = np.dot(gz, self.W1, out=self.gx1)
            gx2 = np.dot(gz, self.W2, out=self.gx2)
            self.gW1 += gz.T.dot(x1)
            self.gW2 += gz.T.dot(x2)
            if not self.nobias:
                self.gb += np.dot(self.gb_ones, gz)
        else:
            gx1 = cp.dot(gz, self.W1, out=self.gx1)
            gx2 = cp.dot(gz, self.W2, out=self.gx2)
            gpu.utils.dot_add(A=gz, B=x1, C=self.gW1, transa=True)
            gpu.utils.dot_add(A=gz, B=x2, C=self.gW2, transa=True)
            if not self.nobias:
                gpu.utils.dot_add(A=self.gb_ones, B=gz, C=self.gb)
         
        return gx1, gx2