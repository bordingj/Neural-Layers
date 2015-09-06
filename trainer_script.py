
from nula.models import BLSTMSequenceEmbed
from document_classifier import DocumentClassifier
import pandas as pd
import string
import sys

if __name__ == '__main__':
    

    sys.stdout.flush()
    
    print('\nThis script trains a Bidirectional LSTM network for sequence embedding/classification on the findZebra corpus')
    
    corpus_path = '/home/bordingj/data/findZebra_corpus.pkl'
    corpus = pd.read_pickle(corpus_path)
    corpus['umls_concept_id'].astype(str, inplace=True)
    corpus['text_lens'] = [len(text) for text in corpus['text']]
    
    no_characters_lower_limit = 200
    corpus = corpus.loc[corpus['text_lens'] >= no_characters_lower_limit,:]
    
    whole_corpus = False
    if not whole_corpus:
        number_of_articles = 30
        assert number_of_articles <= corpus.shape[0]
        corpus = corpus.iloc[:number_of_articles,:]
    
    in_size = len(string.printable)+1
    out_size = corpus['umls_concept_id'].nunique()+1
    
    no_units = 256
    
    dropout_ratio = 0
    
    RNN_model = BLSTMSequenceEmbed(in_size=in_size,
                                  no_labels=out_size, 
                                  no_units=no_units)
                              
    clf = DocumentClassifier(corpus=corpus, dropout_ratio=dropout_ratio,
                                        network=RNN_model)
    
    batchsize     = 400
    seq_len       = 50
    training_time = 2
    no_iterations_per_epoch = 100
    path          = 'BLSTMSequenceEmbed_'
    if dropout_ratio == 0:
        path += 'noDropout_'
    path += 'results.pkl'
    
    print('Starting training ...')
    print('\nTraining BLSTM for sequence classfication/embedding with: \n \
    - no. characters_lower_limit: {0} \n\
     - no. of articles: {1} \n\
     - no. of labels: {2} \n\
     - no. of hidden units: {3} \n\
     - dropout ratio: {4} \n\
     - batchsize: {5} \n\
     - sample sequence length: {6} \n\
     - training time: {7} \n\
     - iterations between monitoring/saving: {8} \n\
     - saving to path: {9} \n\n'.format(
    no_characters_lower_limit, corpus.shape[0], out_size, no_units, dropout_ratio,
    batchsize, seq_len, training_time, no_iterations_per_epoch, path )
    )
    
    clf.fit(path,  batchsize=batchsize, seq_len=seq_len ,
                max_time_in_minutes=training_time,
                no_iterations_per_epoch=no_iterations_per_epoch, 
                clip_threshold=8,
                  on_gpu=True,
                  devices=0,
                  multi_gpu=False)
    
    