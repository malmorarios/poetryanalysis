from gensim.models import Word2Vec

from time import time  # to time our operations
import gzip, json # required to open dataset (gzip) and grab each data point (json) 

import multiprocessing 
cores = multiprocessing.cpu_count() # count the number of cores in a computer (used in creation of Word2Vec model)



## RANDOM LINES OF CODE ##

#import random
#random.sample(all_poetryLines, 8)
#w2v_model.init_sims(replace=True) # makes the model more memory efficient (pre-computes L2 norm)
# run 'w2v_model.wv.vocab' to get a list of the words in the machine's vocabulary


### GLLOBAL VARIABLES ###

create_w2vmodel = False

""" w2v_model = Word2Vec(min_count=2, # ignores all words with total frequency lower than this
                        window=2, # maximum distance between the current and predicted word within a sentence
                        size=300, # dimensionality of word vectors
                        sample=6e-5, # the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5)
                        alpha=0.03, # initial learning rate
                        min_alpha=0.0007, # learning rate will linearly drop to min_alpha as training progresses
                        negative=20, # if >0, negative sampling will be used; specifies how many "noise words" should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
                        workers=cores-1)
 """
# see full list of parameters here: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecVocab

                    

### ----------------- ###


def dataLoader():
    """ loads the Gutenberg Poetry dataset;
    output: a list of all poetry lines loaded """

    print('\n==> Loading poetry line data...')

    limit = 500
    all_dataPoints = []
    for datapoint in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
        all_dataPoints.append(json.loads(datapoint.strip()))
        limit -= 1
        if limit == 0:
            break 
    
    # takes the string parts of the data (the poetry lines) and returns a list of the list of all words in each line of poetry
    all_poetryLines = [datapoint['s'].split() for datapoint in all_dataPoints]

    return all_poetryLines


if __name__ == '__main__':
    
    ## LOADING THE DATA ##
    all_poetryLines = dataLoader()

    if create_w2vmodel:

        ## INITIALIZING MODEL ##
        w2v_model = Word2Vec(min_count=2,
                        window=2,
                        size=300, 
                        sample=6e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=cores-1)
        
        ## BUILDING VOCABULARY ##
        t = time()
        w2v_model.build_vocab(all_poetryLines, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        ## TRAINING WORD VECTORS ##
        t = time()
        w2v_model.train(all_poetryLines, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))



    

   
