from gensim.models import Word2Vec

from time import time 
import gzip, json 

import multiprocessing 
cores = multiprocessing.cpu_count()


### GLLOBAL VARIABLES ###

create_w2vmodel = False
maxpoetrylines = 500

                    
### ----------------- ###


def dataLoader(limit):
    """ loads the Gutenberg Poetry dataset;
    output: a list of all poetry lines loaded """

    print('\n==> Loading poetry line data...\n')

    all_dataPoints = []
    for datapoint in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
        all_dataPoints.append(json.loads(datapoint.strip()))
        limit -= 1
        if limit == 0:
            break 
    
    all_poetryLines = [datapoint['s'].split() for datapoint in all_dataPoints]

    return all_poetryLines


if __name__ == '__main__':
    
    ## LOADING THE DATA ##
    all_poetryLines = dataLoader(maxpoetrylines)

    if create_w2vmodel:

        ## INITIALIZING MODEL ##
        w2v_model = Word2Vec(min_count=2,
                        window=2,
                        size=300, 
                        sample=6e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=cores-1) # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecVocab
        
        ## BUILDING VOCABULARY ##
        t = time()
        w2v_model.build_vocab(all_poetryLines, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        ## TRAINING WORD VECTORS ##
        t = time()
        w2v_model.train(all_poetryLines, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))



    

   
