
import cupy as cp
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
    #remove None
    corpus = corpus.loc[~pd.isnull(corpus['umls_concept_id']),:]
    corpus['umls_concept_id'] = corpus['umls_concept_id'].astype(str)
    corpus['text_lens'] = [len(text) for text in corpus['text']]
    
    #remove very short documents
    no_characters_lower_limit = 200
    corpus = corpus.loc[corpus['text_lens'] >= no_characters_lower_limit,:]
    
    #get disease first section onlye (summary / description)  
    description = corpus.copy()
    description['display_text'] = \
        [text[cut+5:] for text, cut in zip(description['display_text'], 
                                           description['display_text'].str.find('</h3>', 5)) ]
    description['display_text'] = \
        [text[:cut] for text, cut in zip(description['display_text'], 
                                           description['display_text'].str.find('<h3>')) ]
    description['display_text'] = \
        [text[cut+5:] if cut != -1 else text for text, cut in zip(description['display_text'], 
                                           description['display_text'].str.find('</h4>', 5)) ]
    description['display_text'].loc[description['display_text'].str.contains('(#)')] = \
    [text[cut+1:] for text, cut in zip(description['display_text'].loc[description['display_text'].str.contains('(#)')], 
         description['display_text'].loc[description['display_text'].str.contains('(#)')].str.find('\n', 5)) ]
    description['text'] = description['display_text']
    no_characters_lower_limit = 100
    description['text_lens'] = [len(text) for text in description['text']]
    description = description.loc[description['text_lens'] >= no_characters_lower_limit,:]
    
    #get diagnosis and clinical features
    diagnosis = corpus.copy()
    diagnosis['display_text'] = \
        [text[cut+5:] for text, cut in zip(diagnosis['display_text'], 
                                           diagnosis['display_text'].str.find('</h3>', 100)) ]
    diagnosis['display_text'] = \
        [text[cut+5:] if cut != -1 else text for text, cut in zip(diagnosis['display_text'], 
                                           diagnosis['display_text'].str.find('</h4>', 5)) ]
    
    diagnosis['display_text'].loc[diagnosis['display_text'].str.contains('(#)')] = \
         [text[cut+1:] for text, cut in zip(diagnosis['display_text'].loc[diagnosis['display_text'].str.contains('(#)')], 
         diagnosis['display_text'].loc[diagnosis['display_text'].str.contains('(#)')].str.find('\n', 5)) ]
    diagnosis['display_text'].str.replace('<p>','\n')
    diagnosis['display_text'].str.replace('</p>',' ')
    
    diagnosis['text'] = diagnosis['display_text']
    diagnosis['text_lens'] = [len(text) for text in diagnosis['text']]
    diagnosis = diagnosis.loc[diagnosis['text_lens'] >= no_characters_lower_limit,:]
    
    #merge corpus
    corpus = corpus.append(description.append(diagnosis))
    whole_corpus = True
    if not whole_corpus:
        number_of_articles = 50
        assert number_of_articles <= corpus.shape[0]
        corpus = corpus.iloc[:number_of_articles,:]
    
    in_size = len(string.printable)+1
    out_size = corpus['umls_concept_id'].nunique()+1
    
    no_units = 256
    
    dropout_ratio = 0
    with cp.cuda.Device(1):
        RNN_model = BLSTMSequenceEmbed(in_size=in_size,
                                      no_labels=out_size, 
                                      no_units=no_units)
                                  
        clf = DocumentClassifier(corpus=corpus, dropout_ratio=dropout_ratio,
                                            network=RNN_model)
        
        batchsize     = 512
        seq_len       = 100
        training_time = 200*60
        no_iterations_per_epoch = 1000
        path          = 'BLSTMSequenceEmbed_'
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
                      on_gpu=True)
    
    