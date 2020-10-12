import re # provides regular expression matching operations
import os
import random
import numpy as np

# To make sure our kernel runs all the way through and gets saved,
# we'll trim some things back and skip training
IS_KAGGLE = True 

CMU_DICT_PATH = os.path.join(
    '../input', 'cmudict', 'cmudict-0.7b')
CMU_SYMBOLS_PATH = os.path.join(
    '../input', 'cmudict', 'cmudict-0.7b.symbols')

# Skip words with numbers or symbols
#   TODO (10/12/2020): do I want to do this? e.g. "bear" vs. "bear!"
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# Only 3 words are longer than 20 chars
# Setting a limit now simplifies training our model later
#   TODO (10/12/2020): perhaps I want to shorten the min word length to 1?
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2


def load_clean_phonetic_dictionary(): # defining functions inside this function definition means you cannot call them outside of this function!

    def is_alternate_pho_spelling(word):
        # No word has > 9 alternate pronounciations so this is safe
        return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit() # this [word(1) / word(2)] is how the CMU dict notates whether different pronunciations to a word exist (within the dict)

    def should_skip(word):
        if not word[0].isalpha():  # skip symbols (CMU dict defines some punctuation / symbols @beginning before moving on to the word pronunciations)
            return True
        if word[-1] == '.':  # skip abbreviations TODO (10/12/2020): do we want to do this? e.g. "Mr." or "Dr." ?
            return True
        if re.search(ILLEGAL_CHAR_REGEX, word): # determines if the regular expression 'ILLEGAL_CHAR_REGEX' is inside the word 'word'
            return True
        if len(word) > MAX_DICT_WORD_LEN: # self-explanatory
            return True
        if len(word) < MIN_DICT_WORD_LEN: # self-explanatory
            return True
        return False

    phonetic_dict = {}
    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict: # maps all possible byte values in the .txt file to the first 256 Unicode points (basically, if you're opening a .txt file with lots of text, it makes it less error-prone if comp processes the text in this way/encoding (I think))
        for line in cmu_dict:

            # Skip commented lines
            if line[0:3] == ';;;':
                continue

            word, phonetic = line.strip().split('  ')

            # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
            # We don't want to the "(#)" considered as part of the word
            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')]

            if should_skip(word):
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = []
            phonetic_dict[word].append(phonetic)

    if IS_KAGGLE: # limit dataset to 5,000 words
        phonetic_dict = {key:phonetic_dict[key] 
                         for key in random.sample(list(phonetic_dict.keys()), 5000)}
    return phonetic_dict

phonetic_dict = load_clean_phonetic_dictionary()
example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])