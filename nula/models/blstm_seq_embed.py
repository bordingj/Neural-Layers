
import numpy as np
from chainer import cuda, Variable, FunctionSet
import nula.functions as nF
import copy
import pickle

class BLSTMSequenceEmbed(object):
    def __init__(self, in_size, no_labels, no_units, dropout_ratio=0.5):
        """
        Bidirectional Long-short-term memory (LSTM) recurrent neural network, with 2 LSTM layers,
        input to the network is one-hot encoded indices.
        """
        self.prepared = False
        self.function_set = FunctionSet(
            # character embedding
            id_to_h0       = nF.SimpleLayer(in_size=in_size, 
                                             out_size=no_units,
                                             hot=True,
                                            act_func='linear',
                                            nobias=True),
            #forward lstm layers
            h0_to_h1_f       = nF.LSTMLayer(in_size=no_units, 
                                                 out_size=no_units),
            h1_to_h2_f       = nF.LSTMLayer(in_size=no_units, 
                                              out_size=no_units),
            #backward lstm layers
            h0_to_h1_b       = nF.LSTMLayer(in_size=no_units, 
                                                 out_size=no_units),
            h1_to_h2_b       = nF.LSTMLayer(in_size=no_units, 
                                              out_size=no_units),
            #simple layer with 2 inputs
            h2_to_h3          = nF.SimpleLayer2Inputs(in_size=no_units, 
                                                         out_size=no_units,
                                                        act_func='tanh'),
            #output layer
            h_to_y           = nF.SimpleLayer(in_size=no_units, 
                                                    out_size=no_labels,
                                                        act_func='linear'),
            dropout1_f       = nF.Dropout(dropout_ratio, no_units),
            dropout2_f       = nF.Dropout(dropout_ratio, no_units),
            dropout1_b       = nF.Dropout(dropout_ratio, no_units),
            dropout2_b       = nF.Dropout(dropout_ratio, no_units)
        )
        for name, func in self.function_set.__dict__.items():
            setattr(self, name, func)
    

    def prepare(self, T, batchsize, on_gpu):
        for func in self.function_set.__dict__.values():
            func.prepare(batchsize, on_gpu)
        
        functions_visiting_every_timestep = {'id_to_h0': self.id_to_h0,
                                             'dropout1_f': self.dropout1_f,
                                             'dropout2_f': self.dropout2_f,
                                             'dropout1_b': self.dropout1_b,
                                             'dropout2_b': self.dropout2_b,
                                             'h0_to_h1_f': self.h0_to_h1_f,
                                             'h1_to_h2_f': self.h1_to_h2_f,
                                             'h0_to_h1_b': self.h0_to_h1_b,
                                             'h1_to_h2_b': self.h1_to_h2_b}
        assert all([hasattr(self, name) for name in functions_visiting_every_timestep.keys()])
        self.functions_list = [functions_visiting_every_timestep]
        for t in range(T-1):
            new_function_dict = functions_visiting_every_timestep.copy()
            for name, function in new_function_dict.items():
                new_function_dict[name] = function.copy()
            self.functions_list.append(new_function_dict)
        
        self.prepared = True
        
    def set_functions(self, t):
        for name, function in self.functions_list[t].items():
            setattr(self, name, function)

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
                    
    def make_initial_states(self, batchsize, on_gpu=False, train=True):
        if on_gpu:
            module = cuda
        else:
            module = np
        states = {
            'c1': Variable(module.zeros((batchsize, self.h0_to_h1_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'h1': Variable(module.zeros((batchsize, self.h0_to_h1_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'c2': Variable(module.zeros((batchsize, self.h1_to_h2_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'h2': Variable(module.zeros((batchsize, self.h1_to_h2_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train)
        }
        return states
        
    def get_forward_states(self, h0, states_tm1, train=True):
        #forward direction
        if train:
            h1_f, c1_f   = self.h0_to_h1_f(self.dropout1_f(h0, copy_func=not self.prepared), 
                                           states_tm1['h1'], states_tm1['c1'], 
                                        copy_func=not self.prepared)
            h2_f, c2_f   = self.h1_to_h2_f(self.dropout2_f(h1_f, copy_func=not self.prepared),
                                        states_tm1['h2'], states_tm1['c2'], 
                                    copy_func=not self.prepared)
        else:
            h1_f, c1_f   = self.h0_to_h1_f(h0, 
                                           states_tm1['h1'], states_tm1['c1'],
                                             copy_func=not self.prepared)
            h2_f, c2_f   = self.h1_to_h2_f(h1_f,
                                        states_tm1['h2'], states_tm1['c2'],
                                        copy_func=not self.prepared)
        return {'c1': c1_f, 'h1': h1_f, 'c2': c2_f, 'h2': h2_f}
    
    def get_backward_states(self, h0, states_tp1, train=True):
        #backward direction
        h1_b, c1_b   = self.h0_to_h1_b(self.dropout1_b(h0, copy_func=not self.prepared), 
                                       states_tp1['h1'], states_tp1['c1'],
                                        copy_func=not self.prepared)
        h2_b, c2_b   = self.h1_to_h2_b(self.dropout2_b(h1_b, copy_func=not self.prepared),
                                    states_tp1['h2'], states_tp1['c2'],
                                    copy_func=not self.prepared)
        return {'c1': c1_b, 'h1': h1_b, 'c2': c2_b, 'h2': h2_b}
    
    def forward(self, X,  initial_states=None, train=True,
                on_gpu=False):
        """
        Given input batch X this function propagates forward through the network,
        forward and backward through-time and returns a vector representation of the sequences
        In:
            X: 3D array of type int32
                (first dimension must be time, second dimension batch_samples,
                and third index is the size of the input space)
        Out:
            vector representations of all sequences in the batch
        """
        
        if on_gpu:
            X = cuda.to_gpu(X.astype(np.int32))

        T, batchsize, D = X.shape
        assert D == 1
        
        if self.prepared:
            assert T == len(self.functions_list)
            
        if initial_states is None:
            states_tm1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
            states_tp1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        else:
            states_tm1, states_tp1 = initial_states
            if on_gpu:
                for states in (states_tm1, states_tp1):
                    for key, value in states.items():
                        states[key].data = cuda.to_gpu(value.data)
            else:
                for states in (states_tm1, states_tp1):
                    for key, value in states.items():
                        states[key].data = cuda.to_cpu(value.data)
        
        h0_list = []
        #forward through time
        for t in range(T):
            if self.prepared:
                self.set_functions(t)
            x_data = X[t].reshape(batchsize,D)
            x = Variable(x_data, volatile=not train)
            h0 = self.id_to_h0(x, copy_func=not self.prepared)
            states_tm1 = self.get_forward_states(h0, states_tm1, train=train)
            h0_list.append(h0)
        #backward through time
        for t in reversed(range(T)):
            if self.prepared:
                self.set_functions(t)
            states_tp1 = self.get_backward_states(h0_list[t], states_tp1, train=train)
        #get final outputs
        h3 = self.h2_to_h3(states_tm1['h2'], states_tp1['h2'])
        y = self.h_to_y(h3)
        
        return y

    def save(self, fn):
        f = open(fn, 'wb')
        pickle.dump(copy.deepcopy(self).to_cpu(), f); f.close()