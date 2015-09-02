
import pickle
import copy
import numpy as np
from chainer import cuda, Variable, optimizers
import chainer.functions as F
from nula.models import BLSTMSequenceEmbed
from text_handling import corpus_class
import time

#%%

class Model_trainer_predictor(object):
    
    def __init__(self, corpus=None, corpus_path=None, model=None, model_path=None):
        
        if corpus is None and corpus_path is not None:
            with open(corpus_path, 'rb') as f:
                self.corpus = pickle.load(f)
        else:
            self.corpus = corpus
        
        if model is None and model_path is not None:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = model
    
    def save_corpus_and_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'corpus': self.corpus, 
                         'model': copy.deepcopy(self.model).to_cpu()
                         }, f)

    def load_from_CorpusModelDict(self, path):
        with open(path, 'rb') as f:
            CorpusModelDict = pickle.load(f)
            self.model = CorpusModelDict['model']
            self.corpus = CorpusModelDict['corpus']
            
    def get_loss(self, X, labels, on_gpu, print_accuracy=False,
                 initial_states=None):
        
        
        T, batchsize, D = X.shape
        assert labels.shape == (batchsize,)
        
    
            
        y = self.model.forward(X, initial_states=initial_states, train=True,
                    on_gpu=on_gpu)
        
        if print_accuracy:
            probs = F.softmax(y)
            predictions = np.argmax(cuda.to_cpu(probs.data), axis=1)
            accuracy = (predictions == cuda.to_cpu(labels)).sum() / predictions.shape[0]
            print('Accuracy: {}'.format(accuracy))
        
        if on_gpu:
            labels = cuda.to_gpu(labels.astype(np.int32))
            
        t = Variable(labels, volatile=False)
        
        return F.softmax_cross_entropy(y, t)
    
    def train(self, save_path, batchsize, seq_len, no_iterations, clip_threshold=8, 
              on_gpu=True, print_interval=1000):
        
        if on_gpu:
            self.model.to_gpu()
        else:
            self.model.to_cpu()
            
        optimizer = optimizers.Adam()
        optimizer.setup(self.model.function_set)
        
        MyGenerator = self.corpus.SequenceGenerator(seq_len, batchsize, no_iterations,
                               yield_labels=True)
        
        start = time.time()
        
        self.model.prepare(seq_len, batchsize, on_gpu)
        
        for i, (X, labels) in enumerate(MyGenerator):
        
            if i%print_interval==0:
                print_accuracy=True
            else:
                print_accuracy=False
                
            loss = self.get_loss(X, labels, on_gpu=True, print_accuracy=print_accuracy)
            
            optimizer.zero_grads()
            loss.backward()
            optimizer.clip_grads(clip_threshold)
            optimizer.update()
                    
            if i%print_interval==0:
                print('loss at iteration no. {0}: {1}\n'.format(i,loss.data))
                #self.save_corpus_and_model(save_path)
                
        end = time.time()
        print('\nTraining took {} minutes.'.format(int((end-start)/60)))

    def get_probs(self, X, on_gpu):

        y = self.model.forward(X, train=False,
                                on_gpu=on_gpu)
        probs = F.softmax(y)
        
        return cuda.to_cpu(probs.data)

    def get_most_probable(self, model, search_string, counts):
        
        search_string =  " " + search_string + " "
        X = self.corpusGetIndicesFromChars(chars=list(search_string))
        X = X.reshape(X.shape[0],1,1)
        probs = self.get_probs(X=X, on_gpu=True)
        predictions = np.argsort(probs).ravel()[-counts:]
        
        return self.corpus.get_diseases_from_indices(predictions)

#%%
Corpus = corpus_class(
            corpus_path='/home/bordingj/data/findZebra_corpus.pkl', subset_ratio=1/1000)

in_size = len(Corpus.char2index)
out_size = len(Corpus.dis2index)-1
no_units = 500

RNN_model = BLSTMSequenceEmbed(in_size=in_size,
                          no_labels=out_size, 
                          no_units=no_units)
                          
trainer = Model_trainer_predictor(corpus=Corpus, model=RNN_model)

#%%
trainer.train('test.pkl',  batchsize=200, seq_len=50, no_iterations=2000, clip_threshold=8,
              on_gpu=True, print_interval=100)