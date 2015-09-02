
import numpy as np
from chainer import cuda, Variable, FunctionSet
import nula.functions as nF
import chainer.functions as F

class LSTMEmbed(object):
    def __init__(self, in_size, no_labels, no_units, dropout_ratio=0.5):
        #define functions which should be used in model
        self.function_set = FunctionSet(
            # embeddings
            id_to_embed   = nF.SimpleLayer(in_size=in_size, 
                                             out_size=no_units,
                                             hot=True, #input is hot encoded
                                            act_func='linear',
                                            nobias=True),
            #lstm layer
            embed_to_h    = nF.LSTMLayer(in_size=no_units, 
                                                 out_size=no_units),
            #output layer
            h_to_y        = nF.SimpleLayer(in_size=no_units, 
                                                    out_size=no_labels,
                                                        act_func='linear'),
            #dropout function
            dropout       = nF.Dropout(dropout_ratio, no_units),
        )
        for name, func in self.function_set.__dict__.items():
            setattr(self, name, func)
        
        self.prepared = False

    def prepare(self, T, batchsize, on_gpu):
        """
        This function builds a list of length T of dictionaries holding instantiations
        of functions for every timestep
        """
        for func in self.function_set.__dict__.values():
            func.prepare(batchsize, on_gpu)
        
        functions_visiting_every_timestep = {'id_to_embed': self.id_to_embed,
                                             'dropout': self.dropout,
                                             'embed_to_h': self.embed_to_h,
                                             'h_to_y': self.h_to_y}
        self.functions_list = [functions_visiting_every_timestep]
        for t in range(T-1):
            new_function_dict = functions_visiting_every_timestep.copy()
            for name, function in new_function_dict.items():
                new_function_dict[name] = function.copy()
            self.functions_list.append(new_function_dict)
        self.prepared = True
        
    def set_functions(self, t):
        # this function sets the function instances of self to those in self.function_list[t]
        for name, function in self.functions_list[t].items():
            setattr(self, name, function)
                    
    def make_initial_states(self, batchsize, on_gpu=False, train=True):
        if on_gpu:
            module = cuda
        else:
            module = np
        states = {
            'c': Variable(module.zeros((batchsize, self.h0_to_h1_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'h': Variable(module.zeros((batchsize, self.h0_to_h1_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
        }
        return states
    
    def forward_one_step(self, x, states):
        
        embed     = self.id_to_embed(x, copy_func=not self.prepared) 
        h, c   = self.embed_to_h(self.dropout1_b(embed, copy_func=not self.prepared), 
                                  states['h'], states['c'], copy_func=not self.prepared)
        y      = self.h_to_y(h)

        return y, {'c': c, 'h1': h}
        
    def forward(self, X, labels, on_gpu=False):
        """
        Given input batch X this function propagates forward through the network,
        forward through-time and returns the outputs at every time-step
        In:
            X: 3D array of type int32
                (first dimension must be time, second dimension batch_samples,
                and third index is the size of the input space)
            labels: 2d array of type int32
                (first dimension must be time, second dimension batch_samples)
        out:
            cross-entropy loss as chainer-variable
        """
        
        if on_gpu:
            X = cuda.to_gpu(X.astype(np.int32))
            labels = cuda.to_gpu(labels.astype(np.int32)) 
        T, batchsize, D = X.shape
        
        
        if self.prepared:
            assert T == len(self.functions_list)
            
        states = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu)
        
        loss = 0
        #forward through time -recurrently
        for t in range(T):
            
            if self.prepared:# set functions
                self.set_functions(t)
            # propagate one-step forward in-time through network
            y, states = self.get_forward_states(Variable(X[t]), states)
            loss += F.softmax_cross_entropy(y, Variable(labels))
        
        return loss
    
    def to_gpu(self):
        self.function_set.to_gpu()
        if hasattr(self, 'functions_list'):
            for funcs in self.functions_list:
                for func in funcs.values():
                    func.to_gpu()
    
    def to_cpu(self):
        self.function_set.to_cpu()
        if hasattr(self, 'functions_list'):
            for funcs in self.functions_list:
                for func in funcs.values():
                    func.to_cpu()