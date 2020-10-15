import re # provides regular expression matching operations
from pathlib import Path
import random
import numpy as np
import csv

# To make sure our kernel runs all the way through and gets saved,
# we'll trim some things back and skip training
IS_KAGGLE = True 
RECORD_SKIPPEDWORDS = True

CMU_DICT_PATH = Path("input/cmudict/cmudict-0.7b.txt")
#CMU_SYMBOLS_PATH = os.path.join(
    #'../input', 'cmudict', 'cmudict-0.7b.symbols')

# Skip words with numbers or symbols
#   TODO (10/12/2020): do I want to do this? e.g. "bear" vs. "bear!"
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# Only 3 words are longer than 20 chars
# Setting a limit now simplifies training our model later
#   TODO (10/12/2020): perhaps I want to shorten the min word length to 1?
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2


def makeSameLength(LoL):
    """ makes all lists inside a list of lists the same length by continuously appending empty strings """
    maxlength = 0
    for list_obj in LoL:
        if len(list_obj) > maxlength:
            maxlength = list_obj
    
    for list_obj in LoL:
        while len(list_obj) < maxlength:
            list_obj.append("")


def load_clean_phonetic_dictionary(RECORD_SKIPPEDWORDS): # defining functions inside this function definition means you cannot call them outside of this function!

    def is_alternate_pho_spelling(word):
        """ returns True / False if word is an alternate phonetic spelling """
        # No word has > 9 alternate pronounciations so this is safe
        return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit()

    def should_skip(word):
        """ returns True if word should should be skipped; False otherwise """
        if not word[0].isalpha():  # skip symbol + punctuation words and their pronunciations
            return True, 1
        if word[-1] == '.':  # skip abbreviated words TODO (10/12/2020): do we want to do this? e.g. "Mr." or "Dr." ?
            return True, 2
        if re.search(ILLEGAL_CHAR_REGEX, word): # skip word if it has numbers or symbols
            return True, 3
        if len(word) > MAX_DICT_WORD_LEN: # skip word if too long
            return True, 4
        if len(word) < MIN_DICT_WORD_LEN: # skip word if too short
            return True, 5
        return False, 0


    skippedwords_notalpha, skippedwords_abbrev, skippedwords_regex, skippedwords_exceedsmax, skippedwords_lessthanmin = [], [], [], [], []
    phonetic_dict = {}
    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict: 
        for line in cmu_dict:

            # Skip commented lines
            if line[0:3] == ';;;':
                continue
            
            # For every line in the dict, stores the word in that line and its corresponding phonetic pronunciation 
            word, phonetic = line.strip().split('  ') # line.strip(char='') strips away chararacters 'char' from the beginning and end of line; if not given an argument, strips away any white space appearing at the beginning and end of line

            # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
            # We don't want the "(#)" considered as part of the word
            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')] # word.find(substring) returns the index of the FIRST occurence of substring in word. If not found, returns -1. There is also .rfind() which does this but starting from the end of the string!

            if should_skip(word)[0]:
                if RECORD_SKIPPEDWORDS:
                    if should_skip(word)[1] == 0:
                        continue
                    elif should_skip(word)[1] == 1:
                        skippedwords_notalpha.append(word)
                    elif should_skip(word)[1] == 2:
                        skippedwords_abbrev.append(word)
                    elif should_skip(word)[1] == 3:
                        skippedwords_regex.append(word)
                    elif should_skip(word)[1] == 4:
                        skippedwords_exceedsmax.append(word)
                    elif should_skip(word)[1] == 5:
                        skippedwords_lessthanmin.append(word)
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = [] # initializes (key, value) = (word, []) pair in dict. NOTICE you are not appending the pronunciation of the word directly! Just starting with an empty list! See note below.

            # you don't want to be creating different entries in dictionary for every entry 
            #   in CMU dict txt file. By initializing a list as the key for each word and appending the 
            #   pronunciation of that word in this empty list, if the same word shows up in the next line 
            #   with a different pronunciation, you can just NOT create a new entry in the dict and instead 
            #   append the new pronunciation to the existing word's entry in the pronunciation_dict!
            phonetic_dict[word].append(phonetic) 

        if IS_KAGGLE: # limit dataset to 5,000 words
            phonetic_dict = {key:phonetic_dict[key] 
                         for key in random.sample(list(phonetic_dict.keys()), 5000)}

        # recording all skipped words into a .csv file
        if RECORD_SKIPPEDWORDS:
            skippedwords = [skippedwords_notalpha, skippedwords_abbrev, skippedwords_regex, skippedwords_exceedsmax, skippedwords_lessthanmin]
            with open('skippedwords.csv', 'w', newline='') as csvfile:
                fieldnames = ["", "word[0].isalpha() == False", "word[-1] == '.'", "matches REGEX [^A-Z-'.]", "Exceeds max word length", "Less than min word length"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(skippedwords)):
                    for word in skippedwords[i]:
                        writer.writerow({fieldnames[i+1]: word})

    return phonetic_dict

phonetic_dict = load_clean_phonetic_dictionary()
#example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])