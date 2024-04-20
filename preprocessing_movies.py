import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg index->word mapping)
    """
    
    #Possibly manually split the fully connected imdb dataset!! (to make life easier)

    # According to the article:
    # 1. noise removal (special charas + hyperlinks)
    # 2. tokenization & duplicate removal
    #       combine all reviews into one txt file --> maybe do this beforehand?
    #       convert file into list of tokens
    #       remove all duplicates from list and switch to lowercase
    # 3. define the dictionary/vocab size
    #       make a list <key, value> with key = word and value = index of word in list
    #       add <UNK> and digit at the end of the dictionary to acocunt for unk words in future


    # then adress feature vectorization:
    # 1. represent each review as a vector, pulling each word from the vocab through an id
    #         if word unavailable in dictionary, insert <UNK> index into it
    # 2. use lemmatization if word length of review > 25 (process unspecified)
    #         use NLT Python package to remove <STOP> from reviews
    #         use "PCA" for dimensional reduction of feature matrix
    
    
    
    
    # Hint: You might not use all of the initialized variables depending on how you implement preprocessing. 
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []
    
    # 1. convert sentence into concatenated list
    with open(train_file, 'r') as file: # runs through train file
        train_txt = file.read()
    train_data = train_txt.lower().split() # converts it into list of words

    with open(test_file, 'r') as file: # runs through test file
        test_txt = file.read()
    test_data = test_txt.lower().split() # converts it into list of words
      
    # 2. create a list of all unique words
    unique_list = sorted(set(train_data + test_data))

    # 3. create a vocab dictionary that maps each word to its index in the unique list
    vocabulary = {w:i for i, w in enumerate(unique_list)}
    # for word in train_data:
    #     vocabulary[word] = vocabulary.get(word, len(vocabulary))

    # 4. convert into list of tokens (down below)

    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    
    # Uncomment the sanity check below if you end up using vocab_size
    # Sanity check, make sure that all values are withi vocab size
    # assert all(0 <= value < vocab_size for value in vocabulary.values())

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print("train_data", train_data)
    return train_data, test_data, vocabulary