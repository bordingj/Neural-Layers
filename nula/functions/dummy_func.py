
from chainer import function


class DummyFunc(function.Function):

    """Dummy function"""
        
    def forward(self, inputs):
        return inputs
    
    def backward(self, inputs, grad_outputs):
        return grad_outputs
