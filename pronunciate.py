import re # provides regular expression matching operations
from pathlib import Path
import random
import numpy as np
import csv
from pandas import DataFrame
import string


# To make sure our kernel runs all the way through and gets saved,
# we'll trim some things back and skip training
IS_KAGGLE = True 
RECORD_SKIPPEDWORDS = False

# Only 3 words are longer than 20 chars
# Setting a limit now simplifies training our model later
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2

# Relevant paths
CMU_DICT_PATH = Path("input/cmudict/cmudict-0.7b.txt")
CMU_SYMBOLS_PATH = Path("input/cmudict/cmudict-0.7b.symbols.txt")

# Skip words with numbers or symbols
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# We'll need to tell our model where a phonetic spelling starts and ends, 
#   so we'll introduce 2 special start & end symbols, arbitrarily represented by the tab and newline characters.
START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'



def load_clean_phonetic_dictionary():

    # TODO: lol, apparently this process is called "padding" and there is a built in numpy function for it
    def makeSameLength(LoL):
        """ makes all lists inside a list of lists the same length by continuously appending empty strings """
        maxlength = 0
        for list_obj in LoL:
            if len(list_obj) > maxlength:
             maxlength = len(list_obj)
        for list_obj in LoL:
            while len(list_obj) < maxlength:
                list_obj.append("")
        return LoL

    def is_alternate_pho_spelling(word):
        """ returns True / False if word is an alternate phonetic spelling """
        # No word has > 9 alternate pronounciations so this is safe
        return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit()

    def should_skip(word):
        """ returns True if word should should be skipped; False otherwise """
        if not word[0].isalpha():
            return True, 0
        if word[-1] == '.': 
            return True, 1
        if re.search(ILLEGAL_CHAR_REGEX, word):
            return True, 2
        if len(word) > MAX_DICT_WORD_LEN:
            return True, 3
        if len(word) < MIN_DICT_WORD_LEN:
            return True, 4
        return False, -1

    words_notalpha, words_abbrev, words_badregex, words_big, words_small = ([] for i in range(5))
    skippedwords = [words_notalpha, words_abbrev, words_badregex, words_big, words_small]
    phonetic_dict = {}

    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict: 
        for line in cmu_dict:

            # Skip commented lines
            if line[0:3] == ';;;':
                continue
            
            word, phonetic = line.strip().split('  ') # strip() takes away the unecessary white space in the line (e.g. like all the space after the pronunciation)

            # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
            # We don't want the "(#)" considered as part of the word
            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')] # word.find(substring) returns the index of the FIRST occurence of substring in word. If not found, returns -1. There is also .rfind() which does this but starting from the end of the string!

            if should_skip(word)[0]:
                if RECORD_SKIPPEDWORDS:
                    skippedwords[should_skip(word)[1]].append(word)
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = [] # initializes (key, value) = (word, []) pair in dict. NOTICE you are not appending the pronunciation of the word directly! Just starting with an empty list! See note below.

            # you don't want to be creating different entries in dictionary for every entry 
            #   in CMU dict txt file. By initializing a list as the key for each word and appending the 
            #   pronunciation of that word in this empty list, if the same word shows up in the next line 
            #   with a different pronunciation, you can just NOT create a new entry in the dict and instead 
            #   append the new pronunciation to the existing word's entry in the pronunciation_dict!
            phonetic_dict[word].append(phonetic) 
            
        # recording all skipped words into a .csv file
        if RECORD_SKIPPEDWORDS:
            makeSameLength(skippedwords)
            C = {"Starts with Symbol": skippedwords[0],
                    "Abbreviations": skippedwords[1],
                    "Contains Numbers or Symbols": skippedwords[2],
                    "Exceeds 20 Characters": skippedwords[3],
                    "Less than 2 Characters": skippedwords[4],
            }
            df = DataFrame(C, columns= ["Starts with Symbol", "Abbreviations", "Contains Numbers or Symbols", "Exceeds 20 Characters", "Less than 2 Characters"])
            export_csv = df.to_csv (r'skippedwords.csv', index = None, header=True)

        if IS_KAGGLE: # limit dataset to 5,000 words
            phonetic_dict = {key:phonetic_dict[key] for key in random.sample(list(phonetic_dict.keys()), 5000)} # basically creating a random, 5000 length subset of the original dictionary and assigning this subset to the original location in memory of the superset dictionary

    return phonetic_dict


phonetic_dict = load_clean_phonetic_dictionary() # remember: phonetic_dict is a dictionary, where all are words in CMU dictionary and values are a list of the word's pronunciations
example_count = np.sum([len(prons) for prons in phonetic_dict.values()])

print(   "\n".join(  [ wrd+' --> '+phonetic_dict[wrd][0] for wrd in random.sample(list(phonetic_dict.keys()), 10) ]  )   ) 

print('\nAfter cleaning, the dictionary contains %s words and %s pronunciations (%s are alternate pronunciations).' % 
      (len(phonetic_dict), example_count, (example_count-len(phonetic_dict))))


def char_list():
    """ the list of all allowed characters """
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters


def phone_list():
    """ the list of all input phonemes """
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    with open(CMU_SYMBOLS_PATH) as file:
        for line in file: 
            phone_list.append(line.strip())
    return [''] + phone_list 


def id_mappings_from_list(str_list):
    """ assigns an ID to every iterable in str_list; creates itr to ID mappings and ID to itr mappings """
    str_to_id = {s: i for i, s in enumerate(str_list)} 
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str


# Create character to ID mappings (and ID to character mappings); 0th entry is '' string
char_to_id, id_to_char = id_mappings_from_list(char_list()) 

# Create phone to ID mappings (and ID to phone mappings); 0th entry is '' string
phone_to_id, id_to_phone = id_mappings_from_list(phone_list()) 

# Example:
print('Char to id mapping: \n', char_to_id)
print('Phone to id mapping: \n', phone_to_id) # weird -- 0th entry is '' string, but also 1st entry is '\t' and 2nd etnry is '\n'. Phonemes start on 3rd entry?? Why is the start/end phone symbols necessary?

CHAR_TOKEN_COUNT = len(char_to_id)
PHONE_TOKEN_COUNT = len(phone_to_id)


def char_to_1_hot(char):
    """ converts character to 1-hot vector """
    char_id = char_to_id[char]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1.
    return hot_vec


def phone_to_1_hot(phone):
    """ converts phoneme to 1-hot vector """
    phone_id = phone_to_id[phone]
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec


# Example:
print('"A" is represented by:\n', char_to_1_hot('A'), '\n-----')
print('"AH0" is represented by:\n', phone_to_1_hot('AH0'))

MAX_CHAR_SEQ_LEN = max([len(word) for word, _ in phonetic_dict.items()])
MAX_PHONE_SEQ_LEN = max([max([len(pron.split()) for pron in pronuns]) 
                         for _, pronuns in phonetic_dict.items()]
                       ) + 2  # + 2 to account for the start & end tokens we need to add


def dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []
    
    for word, pronuns in phonetic_dict.items():
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT)) # rank 2 tensor (depth of char in char sequence vs. identity of char)
        for i, char in enumerate(word):
            word_matrix[i, :] = char_to_1_hot(char) # this writes the 1-hot vector of char onto the ith row of word_matrix (makes it look like what you thought it would look like [depection in journal -- maybe put this picture in research notes?])
        for pronun in pronuns:
            pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
            pronun_phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
            for i, phone in enumerate(pronun_phones):
                pronun_matrix[i, :] = phone_to_1_hot(phone)
        
            char_seqs.append(word_matrix) # notice that this is adding some words multiple times (per pronunciation it had in the dict)
            phone_seqs.append(pronun_matrix) 
    
    return np.array(char_seqs), np.array(phone_seqs) # completely converts dataset of words and associated pronunciations to an array of word tensors and and an array of pronunciation tensors


char_seq_matrix, phone_seq_matrix = dataset_to_1_hot_tensors()        
print('Word Matrix Shape: ', char_seq_matrix.shape) 
print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape) 


phone_seq_matrix_decoder_output = np.pad( phone_seq_matrix,( (0,0),(0,1),(0,0) ), mode='constant' )[:,1:,:] # padding adds another row of 0s to each phoneme rep; splicing cuts the first row completely from each phoneme rep (eliminating the "start" phoneme token row)


from keras.models import Model
from keras.layers import Input, LSTM, Dense

# TODO: the work required to determine if this is a correct number of hidden nodes for training is NON-TRIVIAL. Look into this for further work.
def baseline_model(hidden_nodes = 256): 

    # Shared Components - Encoder
    char_inputs = Input( shape=(None, CHAR_TOKEN_COUNT) ) # shape: A shape tuple (integers), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.
    encoder = LSTM(hidden_nodes, return_state=True) # return_state: Boolean. Whether to return the last state in addition to the output. Default: False.

    _, state_h, state_c = encoder(char_inputs) # note that the encoder_outputs are discarded; (https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21 explains what the hidden states and cell states are); The LSTM layer in the encoder is defined with the return_state argument set to True. This returns the hidden state output returned by LSTM layers generally, as well as the hidden and cell state for all cells in the layer.
    encoder_states = [state_h, state_c]


    # Shared Components - Decoder
    phone_inputs = Input( shape=(None, PHONE_TOKEN_COUNT) )
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True) # return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.
    decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax') # Prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval {\displaystyle (0,1)}(0,1), and the components will add up to 1, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities.
    
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)


    # Training Model
    training_model = Model([char_inputs, phone_inputs], phone_prediction)

    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)

    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)

    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)

    return training_model, testing_encoder_model, testing_decoder_model

# # TODO: the work required to determine if this is a correct number of hidden nodes for training is NON-TRIVIAL. Look into this for further work.
# def baseline_model(hidden_nodes = 256): 

#     # Shared Components - Encoder
#     char_inputs = Input( shape=(None, CHAR_TOKEN_COUNT) ) # shape: A shape tuple (integers), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.
#     encoder = LSTM(hidden_nodes, return_state=True) # return_state: Boolean. Whether to return the last state in addition to the output. Default: False.

#     # Shared Components - Decoder
#     phone_inputs = Input( shape=(None, PHONE_TOKEN_COUNT) )
#     decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True) # return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.
#     decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax')

#     # Training Model
#     _, state_h, state_c = encoder(char_inputs) # The LSTM layer in the encoder is defined with the return_state argument set to True. This returns the hidden state output returned by LSTM layers generally, as well as the hidden and cell state for all cells in the layer.
#     encoder_states = [state_h, state_c]
#     decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
#     phone_prediction = decoder_dense(decoder_outputs)

#     training_model = Model([char_inputs, phone_inputs], phone_prediction)

#     # Testing Model - Encoder
#     testing_encoder_model = Model(char_inputs, encoder_states)

#     # Testing Model - Decoder
#     decoder_state_input_h = Input(shape=(hidden_nodes,))
#     decoder_state_input_c = Input(shape=(hidden_nodes,))
#     decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
#     decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
#     decoder_states = [decoder_state_h, decoder_state_c]
#     phone_prediction = decoder_dense(decoder_outputs)

#     testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)

#     return training_model, testing_encoder_model, testing_decoder_model


### Training
# First, we'll split off a test set so we can get a fair evaluation 
#   of our model's performance later. For Kaggle, we'll cut the test 
#   size down to just 100 examples.

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
    
(char_input_train, char_input_test, 
 phone_input_train, phone_input_test, 
 phone_output_train, phone_output_test) = train_test_split(
    char_seq_matrix, phone_seq_matrix, phone_seq_matrix_decoder_output, 
    test_size=TEST_SIZE, random_state=42)

TEST_EXAMPLE_COUNT = char_input_test.shape[0]

# Now we'll train our sequence to sequence model until it starts to 
#   overfit. We want a model that generalizes well to previously 
#   unseen examples so we'll keep the version that has the lowest 
#   validation loss.

from keras.callbacks import ModelCheckpoint, EarlyStopping

def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss',patience=3)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=256,
          epochs=100,
          validation_split=0.2, # Keras will automatically create a validation set for us
          callbacks=[checkpointer, stopper])


BASELINE_MODEL_WEIGHTS = Path(
    '/input/predicting-english-pronunciations-model-weights/baseline_model_weights.hdf5')
training_model, testing_encoder_model, testing_decoder_model = baseline_model()
if not IS_KAGGLE:
    train(training_model, BASELINE_MODEL_WEIGHTS, char_input_train, phone_input_train, phone_output_train)