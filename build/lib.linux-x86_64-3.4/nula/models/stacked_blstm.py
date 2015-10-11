
import numpy as np
from chainer import cuda, Variable, FunctionSet
import nula.functions as nF
from collections import deque

class StackedBLSTM(FunctionSet):
    def __init__(self, in_size, no_labels, no_units):
        """
        Bidirectional Long-short-term memory (LSTM) recurrent neural network, with 2 LSTM layers,
        input to the network is one-hot encoded indices.
        """
        super(StackedBLSTM, self).__init__(
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
            #forward lstm layers
            h3_to_h4_f       = nF.LSTMLayer(in_size=no_units, 
                                                 out_size=no_units),
            #backward lstm layers
            h3_to_h4_b       = nF.LSTMLayer(in_size=no_units, 
                                                 out_size=no_units),
                                              
            #simple layer with 2 inputs
            h4_to_h5          = nF.SimpleLayer2Inputs(in_size=no_units, 
                                                         out_size=no_units,
                                                        act_func='tanh'),
            #output layer
            h5_to_y           = nF.SimpleLayer(in_size=no_units, 
                                                    out_size=no_labels,
                                                        act_func='linear'),
            #convex comb
            
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

    def make_initial_states2(self, batchsize, on_gpu=False, train=True):
        if on_gpu:
            module = cuda
        else:
            module = np
        states = {
            'c4': Variable(module.zeros((batchsize, self.h3_to_h4_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'h4': Variable(module.zeros((batchsize, self.h3_to_h4_f.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
        }
        return states
        
    def get_forward_states(self, h0, states_tm1, dropout_ratio, train):
        if dropout_ratio < 0.01:
            train=False
        #forward direction
        h1_f, c1_f   = self.h0_to_h1_f(h0, states_tm1['h1'], states_tm1['c1'])
        h2_f, c2_f   = self.h1_to_h2_f(nF.dropout(h1_f, ratio=dropout_ratio, train=train),
                                    states_tm1['h2'], states_tm1['c2'])
        return {'c1': c1_f, 'h1': h1_f, 'c2': c2_f, 'h2': h2_f}
    
    def get_backward_states(self, h0, states_tp1, dropout_ratio, train):
        if dropout_ratio < 0.01:
            train=False
        #backward direction
        h1_b, c1_b   = self.h0_to_h1_b(h0, states_tp1['h1'], states_tp1['c1'])
        h2_b, c2_b   = self.h1_to_h2_b(nF.dropout(h1_b, ratio=dropout_ratio, train=train),
                                    states_tp1['h2'], states_tp1['c2'])
        return {'c1': c1_b, 'h1': h1_b, 'c2': c2_b, 'h2': h2_b}

    def get_forward_states2(self, h3, states_tm1):
        #forward direction
        h4_f, c4_f   = self.h3_to_h4_f(h3,
                                    states_tm1['h4'], states_tm1['c4'])
        return {'c4': c4_f, 'h4': h4_f}
    
    def get_backward_states2(self, h3, states_tp1):
        #backward direction
        h4_b, c4_b   = self.h3_to_h4_b(h3,
                                    states_tp1['h4'], states_tp1['c4'])
        return {'c4': c4_b, 'h4': h4_b}
    
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
        
        if wPauseMask:
            assert pause_indices is not None
            
        if on_gpu:
            X = cuda.to_gpu(X.astype(np.int32))
            if pause_indices is not None:
                pause_indices = cuda.to_gpu(pause_indices.astype(np.int32))

        T, batchsize, D = X.shape
        assert D == 1
            
        states_tm1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        states_tp1 = self.make_initial_states(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
            
        h0_list = []
        h2_forward_list = []
        #forward through time
        indices_list = []
        for t in range(T):
            x_data = X[t].reshape(batchsize,D)
            if wPauseMask:
                indices = Variable(x_data[:,0], volatile=not train)
                indices_list.append(indices)
                if t > 0:
                    h2_forward_list.append(nF.pauseMask(states_tm1['h2'], indices, 
                                                        pause_indices))
            x = Variable(x_data, volatile=not train)
            h0 = self.id_to_h0(x)
            h0_list.append(h0)
            states_tm1 = self.get_forward_states(h0, states_tm1, 
                                                 dropout_ratio=dropout_ratio,
                                                 train=train)
        h2_forward_list.append(states_tm1['h2'])
    
        #backward through time
        h2_backward_deque = deque([])
        for t in reversed(range(T)):
            indices = indices_list[t]
            if t<(T-1):
                h2_backward_deque.appendleft(nF.pauseMask(states_tp1['h2'], 
                                                          indices, pause_indices))
            states_tp1 = self.get_backward_states(h0_list[t], states_tp1, 
                                                  dropout_ratio=dropout_ratio, 
                                                  train=train)
        h2_backward_deque.appendleft(states_tp1['h2'])
        
        h3_list = []
        for h2_f, h2_b in zip(h2_forward_list, h2_backward_deque):
            h3 = self.h2_to_h3(h2_f, h2_b)
            h3_list.append(h3)
        
        
        states_tm1_2 = self.make_initial_states2(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        states_tp1_2 = self.make_initial_states2(batchsize=batchsize, 
                                                    on_gpu=on_gpu, 
                                                    train=train)
        #forward through time
        for h3 in h3_list:
            states_tm1_2 = self.get_forward_states2(h3, states_tm1_2)
            
        #backward through time
        for h3 in reversed(h3_list):
            states_tp1_2 = self.get_backward_states2(h3, states_tp1_2)
                
            
        #get final outputs
        h5 = self.h4_to_h5(states_tm1_2['h4'], states_tp1_2['h4'])
        y = self.h5_to_y(h5)
        return y

