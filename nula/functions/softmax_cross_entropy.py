import np as np

from chainer import cuda
from chainer import function
from chainer.functions import softmax
from chainer.utils import type_check
import softmax_crossentropy_cpu_funcs

if cuda.available:
    import cupy as cp
    from nula import gpu
    
    @cp.util.memoize(for_each_device=True)
    def _crossentropyloss_kernel():
        kernel_code = cp.carray.compile_with_cache("""
        extern "C" __global__
        void crossentropy(const float *probs, const int *t, float *loss, 
                                      const int N, const int M)
        {   
            __shared__ float shared_vec[16];
            int k;
            int no_whitespaces;
            if (blockIdx.x < N){
            
            for (unsigned int i = threadIdx.x; i < M; i += 16) {
                no_whitespaces = 0;
                for (int k=0; k<no_pause_indices; k++){
                    if (indices[i] == pause_indices[k]){
                        was_a_pause = 1;
                        break;
                    }
                }
                k = t[i];
                logsum += log(fmin(fmax(probs[i * M + k], 1.0E-8), 1.0));
            }
            shared_vec[threadIdx.x] = logsum;
            __syncthreads();
            
        
            if (threadIdx.x == 0) {
                float total_logsum = 0;
                
                for (unsigned int i = 0; i < 32; i++){
                    total_logsum += shared_vec[i];           
                }
                loss[0] = -1.0/N*total_logsum;
                
            }
        }
        }
        """)
        return kernel_code.get_function('crossentropy')
    
    @cp.util.memoize(for_each_device=True)
    def _gsoftmaxCrossentropy_kernel():
        kernel_code = cp.carray.compile_with_cache("""
        extern "C" __global__
        void gsoftmaxCrossentropy(const float *y, const int *t, float coef, 
                                      const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
        
                if (i < N && j < M){
                    if{j == t[i]}{
                        y[i*M+j] -= 1;              
                    }
                    y[i*M+j] *= coef;
                }
        }
        """)
        return kernel_code.get_function('gsoftmaxCrossentropy')
    
class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == np.float32,
            t_type.dtype == np.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        if xp is np:
            self.y, = softmax.Softmax().forward((x,))
            loss = softmax_crossentropy_cpu_funcs.crossentropy(self.y, t)/t.shape[0]
        else:
            self.y, = softmax.Softmax(self.use_cudnn).forward((x,))
            loss = _get_crossentropyloss_gpu(self.y, t)
        return loss.reshape(()),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        gloss = grad_outputs[0]
        
        coef = gloss/x.shape[0]
        if xp is np:
            gx = softmax_crossentropy_cpu_funcs.gsoftmaxCrossentropy(self.y, t, coef)
        else:
            gx = _gsoftmaxCrossentropy_gpu(self.y, t, coef)
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        n_unit = np.prod(self.y.shape[2:], dtype=int)
        if getattr(self, 'normalize', True):
            count = t.shape[0] * n_unit
        else:
            count = t.shape[0]
        coeff = cuda.cupy.divide(gloss, count, dtype=gloss.dtype)
        gx = cuda.elementwise(
            'T y, raw S t, raw T coeff, S n_channel, S n_unit',
            'T gx',
            '''
               const int n = i / (n_channel * n_unit);
               const int c = (i % (n_channel * n_unit)) / n_unit;
               const int m = i % n_unit;
               gx = coeff[0] * (y - (c == t[n * n_unit + m]));
            ''',
            'softmax_crossent_bwd')(
                self.y, t, coeff, self.y.shape[1], n_unit)
        return gx, None


def softmax_cross_entropy(x, t, use_cudnn=True, normalize=True):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (Variable): Variable holding a multidimensional array whose element
            indicates unnormalized log probability: the first axis of the
            variable represents the number of samples, and the second axis
            represents the number of classes. While this function computes
            a usual softmax cross entropy if the number of dimensions is equal
            to 2, it computes a cross entropy of the replicated softmax if the
            number of dimensions is greater than 2.
        t (Variable): Variable holding an int32 vector of groundtruth labels.
        normalize (Variable): Variable holding a boolean value which
            determines the normalization constant. If true, this function
            normalizes the cross entropy loss across all instances. If else,
            it only normalizes along a batch size.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SoftmaxCrossEntropy(use_cudnn)(x, t)
    



def _get_crossentropyloss_gpu(probs, t):
    
    
    kernel = _crossentropyloss_kernel()
    N, M = probs.shape
    loss = cp.empty((1,), dtype=np.float32)
    kernel(grid=(N, 1, 1), block=(32, 1, 1),
             args=(probs, t, loss,
                   np.int32(N),
                   np.int32(M)
                   ) )
    return loss

def _gsoftmaxCrossentropy_gpu(y, t, coef):
    
    
    kernel = _gsoftmaxCrossentropy_kernel()
    N, M = y.shape
    if N == 1:
        bdim, gdim = gpu.utils.Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = gpu.utils.Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = gpu.utils.Get_bdim_and_gdim2D(N,M) 
    
    kernel(grid=gdim, block=bdim,
             args=(y, t, coef,
                   np.int32(N),
                   np.int32(M)
                   ) )
    return y