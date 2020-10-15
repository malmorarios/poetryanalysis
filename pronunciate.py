import re # provides regular expression matching operations
from pathlib import Path
import random
import numpy as np
import csv
from pandas import DataFrame

# To make sure our kernel runs all the way through and gets saved,
# we'll trim some things back and skip training
IS_KAGGLE = True 
RECORD_SKIPPEDWORDS = True

CMU_DICT_PATH = Path("input/cmudict/cmudict-0.7b.txt")

# Skip words with numbers or symbols
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# Only 3 words are longer than 20 chars
# Setting a limit now simplifies training our model later
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2


def load_clean_phonetic_dictionary():

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
            phonetic_dict = {key:phonetic_dict[key] for key in random.sample(list(phonetic_dict.keys()), 5000)}

    return phonetic_dict
        
phonetic_dict = load_clean_phonetic_dictionary()
#example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])