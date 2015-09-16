
import numpy as np
from chainer import cuda, Variable, FunctionSet
import nula.functions as nF
from collections import deque

class BLSTM(FunctionSet):
    def __init__(self, in_size, no_labels, no_units):
        """
        Bidirectional Long-short-term memory (LSTM) recurrent neural network, with 2 LSTM layers,
        input to the network is one-hot encoded indices.
        """
        super(BLSTM, self).__init__(
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
        )
                    
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
        
    def get_forward_states(self, h0, states_tm1, dropout_ratio, train=True):
        if dropout_ratio < 0.01:
            train=False
        #forward direction
        h1_f, c1_f   = self.h0_to_h1_f(h0, states_tm1['h1'], states_tm1['c1'])
        h2_f, c2_f   = self.h1_to_h2_f(nF.dropout(h1_f, ratio=dropout_ratio, train=train),
                                    states_tm1['h2'], states_tm1['c2'])
        return {'c1': c1_f, 'h1': h1_f, 'c2': c2_f, 'h2': h2_f}
    
    def get_backward_states(self, h0, states_tp1, dropout_ratio, train=True):
        if dropout_ratio < 0.01:
            train=False
        #backward direction
        h1_b, c1_b   = self.h0_to_h1_b(h0, states_tp1['h1'], states_tp1['c1'])
        h2_b, c2_b   = self.h1_to_h2_b(nF.dropout(h1_b, ratio=dropout_ratio, train=train),
                                    states_tp1['h2'], states_tp1['c2'])
        return {'c1': c1_b, 'h1': h1_b, 'c2': c2_b, 'h2': h2_b}
    
    def forward(self, X, dropout_ratio, train=True, on_gpu=False,
                wPauseMask=False, pause_indices=None):
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
            
        states_tm1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        states_tp1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        
        h0_list = []
        #forward through time
        for t in range(T):
            x_data = X[t].reshape(batchsize,D)
            x = Variable(x_data, volatile=not train)
            h0 = self.id_to_h0(x)
            states_tm1 = self.get_forward_states(h0, states_tm1, 
                                                 dropout_ratio=dropout_ratio,
                                                 train=train)
            h0_list.append(h0)
        #backward through time
        for t in reversed(range(T)):
            states_tp1 = self.get_backward_states(h0_list[t], states_tp1, 
                                                  dropout_ratio=dropout_ratio, 
                                                  train=train)
        #get final outputs
        h3 = self.h2_to_h3(states_tm1['h2'], states_tp1['h2'])
        y = self.h_to_y(h3)
        return y
    
class CharBLSTM(FunctionSet):
    def __init__(self, in_size, no_labels, no_units):
        """
        Bidirectional Long-short-term memory (LSTM) recurrent neural network, with 2 LSTM layers,
        input to the network is one-hot encoded indices.
        """
        super(CharBLSTM, self).__init__(
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
            #autoclassifier
            h2_to_y     = nF.SimpleLayer2Inputs(in_size=no_units, 
                                                         out_size=in_size,
                                                        act_func='linear')
        )
                    
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
        
    def get_forward_states(self, h0, states_tm1, dropout_ratio, train=True):
        if dropout_ratio < 0.01:
            train=False
        #forward direction
        h1_f, c1_f   = self.h0_to_h1_f(h0, states_tm1['h1'], states_tm1['c1'])
        h2_f, c2_f   = self.h1_to_h2_f(nF.dropout(h1_f, ratio=dropout_ratio, train=train),
                                    states_tm1['h2'], states_tm1['c2'])
        return {'c1': c1_f, 'h1': h1_f, 'c2': c2_f, 'h2': h2_f}
    
    def get_backward_states(self, h0, states_tp1, dropout_ratio, train=True):
        if dropout_ratio < 0.01:
            train=False
        #backward direction
        h1_b, c1_b   = self.h0_to_h1_b(h0, states_tp1['h1'], states_tp1['c1'])
        h2_b, c2_b   = self.h1_to_h2_b(nF.dropout(h1_b, ratio=dropout_ratio, train=train),
                                    states_tp1['h2'], states_tp1['c2'])
        return {'c1': c1_b, 'h1': h1_b, 'c2': c2_b, 'h2': h2_b}
    
    def forward(self, X, dropout_ratio, train=True, on_gpu=False, return_targets=True,
                return_final_hidden_states=False):
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
            
        states_tm1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        states_tp1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        
        h0_list = []
        #forward through time
        h2_forward_list = []
        for t in range(T):
            x_data = X[t].reshape(batchsize,D)
            x = Variable(x_data, volatile=not train)
            h0 = self.id_to_h0(x)
            states_tm1 = self.get_forward_states(h0, states_tm1, 
                                                 dropout_ratio=dropout_ratio,
                                                 train=train)
            h0_list.append(h0)
            h2_forward_list.append(states_tm1['h2'])
            
        #backward through time
        h2_backward_deque = deque([])
        for t in reversed(range(T)):
            states_tp1 = self.get_backward_states(h0_list[t], states_tp1, 
                                                  dropout_ratio=dropout_ratio, 
                                                  train=train)
            h2_backward_deque.appendleft(states_tp1['h2'])
        
        if return_final_hidden_states:
            return states_tm1['h2'], states_tp1['h2']
            
        #get autoregressive outputs
        y_list = []
        for t in range(T-2):
            h2_f = h2_forward_list[t]
            h2_b = h2_backward_deque[t+2]
            y = self.h2_to_y(h2_f, h2_b)
            y_list.append(y)
        if return_targets:
            targets_list = []
            for t in range(T-1):
                x_data = X[t+1].ravel()
                targets = Variable(x_data, volatile=not train)
                targets_list.append(targets)
            return y_list, targets_list
        return y_list
        