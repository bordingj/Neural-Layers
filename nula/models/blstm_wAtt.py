
import numpy as np
from chainer import cuda, Variable, FunctionSet
import nula.functions as nF
import chainer.functions as F
from collections import deque

if cuda.available:
    import cupy as cp

def dummy_func(x):
    return x
    
class BLSTMwAtt(FunctionSet):
    def __init__(self, in_size, no_icd10_block, no_supericd10_code, 
                         no_icd10_cui_mix, no_icd10_code, no_cui,
                 no_units, dropout, dropout_ratio=0.5):
        """
        Bidirectional Long-short-term memory (LSTM) recurrent neural network, with 2 LSTM layers,
        input to the network is one-hot encoded indices.
        """
        
        if dropout_ratio < 0.01:
            dropout=False
        if dropout:
            no_units_secondlstm_layer = int(1/(1-dropout_ratio)*no_units)
        else:
            no_units_secondlstm_layer = no_units
        word_level_no_units = no_units_secondlstm_layer*2
        
        super(BLSTMwAtt, self).__init__(
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
                                                  out_size=no_units_secondlstm_layer),
                
            #backward lstm layers
            h0_to_h1_b       = nF.LSTMLayer(in_size=no_units, 
                                                 out_size=no_units),

            h1_to_h2_b       = nF.LSTMLayer(in_size=no_units, 
                                              out_size=no_units_secondlstm_layer),
                                              
            #simple layer with 2 inputs
            h2_to_h3          = nF.SimpleLayer2Inputs(in_size=no_units_secondlstm_layer, 
                                                         out_size=word_level_no_units,
                                                        act_func='tanh'),
            #attention functions
            h3_to_a           = nF.SimpleLayer(in_size=word_level_no_units,
                                               out_size=1,
                                               act_func='linear',
                                               nobias=True),
                                               
            c_to_h4           = nF.SimpleLayer(in_size=word_level_no_units,
                                               out_size=word_level_no_units,
                                               act_func='leakyrelu'),
            #output layers
            h4_to_icd10_block           = nF.SimpleLayer(in_size=word_level_no_units, 
                                               out_size=no_icd10_block,
                                               act_func='linear'),
            h4_to_supericd_code           = nF.SimpleLayer(in_size=word_level_no_units, 
                                               out_size=no_supericd10_code,
                                               act_func='linear'),
            h4_to_icd10_cui_mix           = nF.SimpleLayer(in_size=word_level_no_units, 
                                               out_size=no_icd10_cui_mix,
                                               act_func='linear'),
            h4_to_icd10_code           = nF.SimpleLayer(in_size=word_level_no_units, 
                                               out_size=no_icd10_code,
                                               act_func='linear'),
            h4_to_cui           = nF.SimpleLayer(in_size=word_level_no_units, 
                                               out_size=no_cui,
                                               act_func='linear'),
                                               
            dropout        = nF.Dropout(dropout_ratio) if dropout else dummy_func
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
        
    def get_forward_states(self, h0, states_tm1, train):
        #forward direction
        h1_f, c1_f   = self.h0_to_h1_f(h0, states_tm1['h1'], states_tm1['c1'])
        if train:
            h2_f, c2_f   = self.h1_to_h2_f(self.dropout(h1_f),
                                        states_tm1['h2'], states_tm1['c2'])
        else:
            h2_f, c2_f   = self.h1_to_h2_f(h1_f,
                                        states_tm1['h2'], states_tm1['c2'])
        return {'c1': c1_f, 'h1': h1_f, 'c2': c2_f, 'h2': h2_f}
    
    def get_backward_states(self, h0, states_tp1, train):
        #backward direction
        h1_b, c1_b   = self.h0_to_h1_b(h0, states_tp1['h1'], states_tp1['c1'])
        if train:
            h2_b, c2_b   = self.h1_to_h2_b(self.dropout(h1_b),
                                        states_tp1['h2'], states_tp1['c2'])
        else:
            h2_b, c2_b   = self.h1_to_h2_b(h1_b,
                                        states_tp1['h2'], states_tp1['c2'])
        return {'c1': c1_b, 'h1': h1_b, 'c2': c2_b, 'h2': h2_b}
    
    def get_context(self, h2_forward_list, h2_backward_deque, train):
        xp = np if isinstance(h2_forward_list[0].data, np.ndarray) else cp
            
        c = Variable(xp.zeros((h2_forward_list[0].data.shape[0], self.h3_to_a.in_size),
                               dtype=np.float32), volatile=not train)
                     
        for i, (h2_f, h2_b) in enumerate(zip(h2_forward_list, h2_backward_deque)):
            h3 = self.h2_to_h3(h2_f, h2_b)
            if train:
                a  = F.softmax(
                            self.h3_to_a( self.dropout(h3)
                            )
                        )
            else:
                a  = F.softmax(
                            self.h3_to_a( h3
                            )
                        )
            c  = nF.addMatVecElementwiseProd(a, h3, c)
        return c
                
    def forward(self, X, whitespace_indices, train, on_gpu, ID_col, **kwargs):
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
            whitespace_indices = cuda.to_gpu(whitespace_indices.astype(np.int32))

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
        inputs_list = []
        for t in range(T):
            x_data = X[t].reshape(batchsize,D)
            x = Variable(x_data, volatile=not train)
            indices = Variable(x_data[:,0], volatile=not train)
            indices_list.append(indices)
            inputs_list.append(x)
            h0 = self.id_to_h0(x)
            states_tm1 = self.get_forward_states(h0, states_tm1, 
                                                 train=train)
            h0_list.append(h0)

            if t < (T-1):
                h2_forward_list.append(nF.pauseMask(states_tm1['h2'], indices, 
                                                         whitespace_indices))
            else:
                h2_forward_list.append(states_tm1['h2'])

        if len(h2_forward_list) == 0:
            h2_forward_list.append(states_tm1['h2'])
    
        #backward through time
        h2_backward_deque = deque([])
        for t in reversed(range(T)):
            x = inputs_list[t]
            indices = indices_list[t]
            states_tp1 = self.get_backward_states(h0_list[t], states_tp1, 
                                                  train=train)
            if t > 0:
                h2_backward_deque.appendleft(nF.pauseMask(states_tp1['h2'], indices, 
                                                               whitespace_indices))
            else:
                h2_backward_deque.appendleft(states_tp1['h2'])

        if len(h2_backward_deque) == 0:
            h2_backward_deque.appendleft(states_tp1['h2'])
                    
        #attention function / context
        c = self.get_context(h2_forward_list, h2_backward_deque, 
                             train=train)
        #get final outputs
        h4  = self.c_to_h4(c)
        output_func = getattr(self, 'h4_to_'+ID_col)
        y   = output_func(h4)
        return y