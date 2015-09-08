
import numpy as np
import pandas as pd
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import time
import string
import random
import pickle
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def accuracy(probs, t):
    pred = np.argmax(probs, axis=1).astype(np.int32)
    assert pred.dtype == t.dtype
    assert pred.shape == t.shape
    return (pred == t).sum() / t.shape[0]

def Recallatk(probs, t, k):
    pred = np.argsort(probs, axis=1).astype(np.int32)
    assert k <= pred.shape[1]
    assert pred.dtype == t.dtype
    relevant_docs = np.zeros_like(t).astype(np.bool)
    for i in range(1,k+1):
        predictions = pred[:,-i].ravel()
        assert t.shape == predictions.shape
        relevant_docs  += (predictions == t)
    relevant_docs = relevant_docs.astype(np.float32)
    return relevant_docs.mean()



class DocumentClassifier(object):
    
    def __init__(self, network, corpus, dropout_ratio):
        
        self.dropout_ratio = dropout_ratio
        self.char2idx = {char: idx for idx, char in enumerate(string.printable)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        idx = len(self.idx2char)
        self.idx2char[idx] = 'Ø'
        self.char2idx['Ø'] = idx
        
        self.corpus  = corpus
        self.ID2idx   = {ID: index for index, ID in enumerate(self.corpus['umls_concept_id'].unique())}
        self.ID2idx['UNKNOWN'] = len(self.ID2idx)
        self.idx2ID  = {idx: ID for ID, idx in self.ID2idx.items()}
        self.network = network
        
        self.train_performance = {}
        self.test_performance = {}
        
    def save(self, path):
        with open(path,'wb') as f:
            a_copy = copy.copy(self)
            setattr(a_copy, 'network', copy.deepcopy(self.network).to_cpu() )
            pickle.dump(a_copy, f)

    def GetIndexFromChar(self, char):
        try:
            return self.char2idx[char]
        except:
            #print('Got an unseen character: {0}'.format(char))
            return len(self.char2idx)
                
    def GetIndicesFromChars(self, chars):
        """
        In:
            chars: iterable of characters
        Out:
            1D numpy array of int32 with corresponding indices
        """
            
        return np.array([self.GetIndexFromChar(char) for char in chars], dtype=np.int32)

    def GetIndexFromID(self, ID):
        try:
            return self.ID2idx[ID]
        except:
            #print('Got an unseen id: {0}'.format(ID))
            return len(self.ID2idx)
                
    def GetIndicesFromIDs(self, IDs):
        """
        In:
            IDs: iterable of IDs
        Out:
            1D numpy array of int32 with corresponding indices g
        """
        def GetIndexFromID(ID):
            try:
                return self.ID2idx[ID]
            except:
                print('Got an unseen id: {0}'.format(ID))
                return len(self.ID2idx)
            
        return np.array([self.GetIndexFromID(ID) for ID in IDs], dtype=np.int32)

    def SequenceGenerator(self, seq_len, batchsize, no_iterations):
        """
        seq_len (int): length of each sample in minibatch
        batchsize (int): size of minibatch
        no_iterations (int): length of the generator
        """
        num_articles = self.corpus.shape[0]
        
        for k in range(no_iterations):
            X = np.empty((seq_len, batchsize), dtype=np.int32)
            labels = np.empty((batchsize,), dtype=np.int32)
            texts_indices = np.random.randint(0, num_articles, batchsize)
            for i, text_index in enumerate(texts_indices):
                if self.corpus['text_lens'].iloc[text_index] > seq_len:
                    starting_index = random.randint(0, (self.corpus['text_lens'].iloc[text_index]-seq_len))
                else:
                    starting_index = 0
                text = self.corpus['text'].iloc[text_index][starting_index:starting_index+seq_len]
                
                X[:,i] =  self.GetIndicesFromChars(list(text))
                labels[i] = self.ID2idx[self.corpus['umls_concept_id'].iloc[text_index]]
            
            X = np.atleast_3d(X)
            yield X, labels

    def get_loss(self, X, labels, on_gpu):
                         
        T, batchsize, D = X.shape
        assert labels.shape == (batchsize,)
        
        y = self.network.forward(X,
                                 dropout_ratio=self.dropout_ratio, 
                                 train=True,
                                 on_gpu=on_gpu)
            
        if on_gpu:
            labels = cuda.to_gpu(labels.astype(np.int32))
                
        t = Variable(labels, volatile=False)
            
        return F.softmax_cross_entropy(y, t)
    
    def predict(self, X, on_gpu):
        T, batchsize, D = X.shape
        
        y = self.network.forward(X, 
                                 dropout_ratio=self.dropout_ratio, 
                                 train=False,
                                 on_gpu=on_gpu)
        
        return cuda.to_cpu(F.softmax(y).data).copy()

    def fit(self, save_path, batchsize, seq_len, 
                    max_time_in_minutes, no_iterations_per_epoch=1000, 
                    clip_threshold=8, on_gpu=True):
        
        if on_gpu:
            self.network.to_gpu()
        else:
            self.network.to_cpu()
        
        
        optimizer = optimizers.Adam()
        optimizer.setup(self.network)

        train_loss_list = []
        train_accuracy_list = []
        train_Recallat5_list = []
        train_Recallat10_list = []
        train_Recallat20_list = []
        test_accuracy_list = []
        test_Recallat5_list = []
        test_Recallat10_list = []
        test_Recallat20_list = []
        start = time.time()
        time_elapsed = 0
        max_time = max_time_in_minutes*60
        i = 0
        k = 0
        while max_time > time_elapsed:
            
            train_Generator = self.SequenceGenerator(seq_len, batchsize, 
                                                     no_iterations_per_epoch)
            for X, labels in train_Generator:
                k += 1
                
                optimizer.zero_grads() #important! before anything!
                loss = self.get_loss(X, labels, 
                                    on_gpu=on_gpu)
                                    
                loss.backward()                
                optimizer.clip_grads(clip_threshold)
                optimizer.update()
            
            #get train performance
            train_loss_list.append(cuda.to_cpu(loss.data).copy())
            probs = self.predict(X, on_gpu)
            train_accuracy_list.append(accuracy(probs, labels))
            train_Recallat5_list.append(Recallatk(probs, labels, k=5))
            train_Recallat10_list.append(Recallatk(probs, labels, k=10))
            train_Recallat20_list.append(Recallatk(probs, labels, k=20))
            print('\nLoss at iteration no. {0}: {1}'.format(k, train_loss_list[i]))
            print('Train Accuracy: {0:3.2f}'.format(train_accuracy_list[i]))
            print('Train Recall@5: {0:3.2f}'.format(train_Recallat5_list[i]))
            print('Train Recall@10: {0:3.2f}'.format(train_Recallat10_list[i]))
            print('Train Recall@20: {0:3.2f}\n'.format(train_Recallat20_list[i]))
            self.train_performance = {'train_loss': train_loss_list,
                                      'train_accuracy': train_accuracy_list,
                                      'train_Recallat5_list': train_Recallat5_list,
                                      'train_Recallat10_list': train_Recallat10_list,
                                      'train_Recallat20_list': train_Recallat20_list
                                        }
            #get test performance
            test_Generator = self.SequenceGenerator(seq_len, batchsize, 1)
            X, labels = next(test_Generator)
            probs = self.predict(X, on_gpu)
            test_accuracy_list.append(accuracy(probs, labels))
            test_Recallat5_list.append(Recallatk(probs, labels, k=5))
            test_Recallat10_list.append(Recallatk(probs, labels, k=10))
            test_Recallat20_list.append(Recallatk(probs, labels, k=20))
            print('Test Accuracy: {0:3.2f}'.format(test_accuracy_list[i]))
            print('Test Recall@5: {0:3.2f}'.format(test_Recallat5_list[i]))
            print('Test Recall@10: {0:3.2f}'.format(test_Recallat10_list[i]))
            print('Test Recall@20: {0:3.2f}\n'.format(test_Recallat20_list[i]))
            self.test_performance = {'test_accuracy': test_accuracy_list,
                                      'test_Recallat5_list': test_Recallat5_list,
                                      'test_Recallat10_list': test_Recallat10_list,
                                      'test_Recallat20_list': test_Recallat20_list
                                        }
            
            
            
            self.save(save_path)
            time_elapsed = time.time()-start
            i += 1
            
            pd.DataFrame(data={'test_accuracy': test_accuracy_list, 
                        'train_accuracy': train_accuracy_list}).plot()
            plt.savefig(str(type(self.network))[9:]+'_accuracy_'+'.png', format='png')
            plt.close()
        print('\nTraining took {} minutes.'.format(int(time_elapsed/60)))
        print('Performed {} iterations.'.format(k))
        print('Saw {} samples per second'.format(batchsize*k/time_elapsed))
        self.save(save_path)
        
    def predict_single_string(self, search_string, on_gpu, return_as_IDs=True):
        
        search_string =  " " + search_string + " "
        X = self.GetIndicesFromChars(chars=list(search_string))
        X = X.reshape(X.shape[0],1,1)
        
        probs = self.predict(X, on_gpu)
        predictions = np.argsort(probs.ravel())
        
        if return_as_IDs:
            return [self.idx2ID[idx] for idx in reversed(predictions)]
        else:
            return np.flipud(predictions)