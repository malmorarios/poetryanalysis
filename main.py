import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing

import gzip, json # required to open dataset (gzip) and grab each data point (json) 

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

import multiprocessing
from gensim.models import Word2Vec

import random

cores = multiprocessing.cpu_count() # Count the number of cores in a computer


### GLLOBAL VARIABLES ###

# total amount of poetry lines loaded from Gutenberg dataset
w2v_model = Word2Vec(min_count=2,
                        window=2,
                        size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)

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
    t = time()

    ## LOADING THE DATA ##
    all_poetryLines = dataLoader()

    random.sample(all_poetryLines, 8)

    ## BUILDING WORD2VEC MODEL ##
    w2v_model.build_vocab(all_poetryLines, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    w2v_model.train(all_poetryLines, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True) # makes the model more memory efficient (pre-computes L2 norm)

   
