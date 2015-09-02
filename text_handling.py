
import pandas as pd
import numpy as np
import pickle
import random

class corpus_class(object):
    def __init__(self, corpus=None, corpus_path=None, no_characters_lower_limit=200,
                 subset_ratio=None, unseen_tag='(UNSEEN)',
                char_count_cutoff=1000,
                character_dicts = None, disease_dicts = None):
        """
        In:
            corpus_path: path to pickle serialized pandas dataframe
            no_characters_lower_limit: will remove texts with characters less than this limit
            subset_ratio: will only the first subset of texts subset_ratio of the corpus
            unseen_tag: tag for an unseen character in character2index map
            char_count_cutoff: will only consider characters that appear more than this number of times in corpus texts
        """
        if corpus is None and corpus_path is None:
            raise NameError('corpus and corpus_path cannot both be none')
            
        if corpus is None:
            self.corpus = pd.read_pickle(corpus_path)
            self.corpus['text_lens'] = [len(text) for text in self.corpus['text']]
            self.corpus = self.corpus.loc[self.corpus['text_lens'] >= no_characters_lower_limit,:]
            
            if subset_ratio is not None:
                self.corpus = self.corpus.iloc[:int(self.corpus.shape[0]*subset_ratio),:]
        else:
            self.corpus = corpus
                
        if character_dicts is None:
            self.char2index, self.index2char = self._Get_index2char_and_char2index_dicts(char_count_cutoff)
            # add an index for an unseen char
            idx = len(self.index2char)
            self.index2char[idx] = unseen_tag
            self.char2index[unseen_tag] = idx
            self.unseen_tag = unseen_tag
        else:
            self.char2index, self.index2char = character_dicts
            self.unseen_tag = self.index2char[len(self.index2char)]
            
        if disease_dicts is None:
            # dictionary-maps for each disease
            self.dis2index = dict(zip(set(self.corpus['disease_name']), list(range(len(set(self.corpus['disease_name']))))))
            self.index2dis = dict(zip(list(range(len(set(self.corpus['disease_name'])))), set(self.corpus['disease_name'])))
            # add an index for an blank label
            idx = len(self.index2dis)
            self.index2dis[idx] = 'blank'
            self.dis2index['blank'] = idx
        else:
            self.dis2index, self.index2dis = disease_dicts
            

        
    def _Get_index2char_and_char2index_dicts(self, char_count_cutoff):
        """
        In: 
            Texts_list: iteratble of texts
            unseen_tag (string): tag for an unseen character
            end_of_doc_tag (string): tag for end of document
        Out: 
            index2char (dict): dictionary which maps an integer to a character
            char2index (dict): dictionary which maps a character to an integer
        """
        char_counts = {}
        #iterate through all texts
        for i, text in enumerate(self.corpus['text']):
            #add counts
            for char in set(text):
                try:
                    char_counts[char] += text.count(char)
                except:
                    char_counts[char] = text.count(char)
    
        all_chars = {char if value >= char_count_cutoff else None for char, value in char_counts.items()}
        all_chars -= {None}
        all_chars = list(all_chars)
        # make char2index and index2char dictionaries
        char2index = dict(zip(all_chars, list(range(len(all_chars)))))
        index2char = dict(zip(list(range(len(all_chars))), all_chars))
        
        return char2index, index2char
    
    def GetIndicesFromChars(self, chars):
        """
        In:
            chars: iterable of characters
        Out:
            1D numpy array of int32 with corresponding indices with appended end-of-document tag
        """
        def GetIndexFromChar(char):
            try:
                index = self.char2index[char]
            except:
                index = self.char2index[self.unseen_tag]
            return index
            
        indices = list(map(lambda x: GetIndexFromChar(x), chars))
        return np.array(indices, dtype=np.int32)

    def SequenceGenerator(self, seq_len, batchsize, no_iterations, 
                              yield_labels=True):
        """
        seq_len (int): length of each sample in minibatch
        batchsize (int): size of minibatch
        no_iterations (int): length of the generator
        yield_labels (bool): if true the generator will yield labels (disease names)
        """
        num_articles = self.corpus.shape[0]
        
        for k in range(no_iterations):
            X = np.empty((seq_len, batchsize), dtype=np.int32)
            texts_indices = np.random.randint(0, num_articles, batchsize)
            labels = np.empty((batchsize,), dtype=np.int32)
            for i, text_index in enumerate(texts_indices):
                if self.corpus['text_lens'].iloc[text_index] > seq_len:
                    starting_index = random.randint(0, (self.corpus['text_lens'].iloc[text_index]-seq_len))
                else:
                    starting_index = 0
                text = self.corpus['text'].iloc[text_index][starting_index:starting_index+seq_len]
                
                data = self.GetIndicesFromChars(chars=list(text))
                X[:,i] = data
                
                labels[i] = self.dis2index[self.corpus['disease_name'].iloc[text_index]]
            X = np.ascontiguousarray(np.atleast_3d(X))
            if yield_labels:
                yield X, labels
            else:
                yield X
    
    def get_diseases_from_indices(self, indices):

        return list(map(lambda x: self.index2dis[x], indices))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)