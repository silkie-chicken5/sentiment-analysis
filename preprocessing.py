import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

imdb_dict = None
airline_dict = None
election_dict = None
anime_dict = None

def preprocess():
    'Preprocesses the data from all four datasets. Sets the lemmatization and UNKing parameters'
    imdb_dict = preprocess_data('data/imdb_reviews.csv', 'review', 'sentiment', 25, 15)
    airline_dict = preprocess_data('data/airline_tweets.csv', 'text', 'airline_sentiment', 20, 5)
    election_dict = preprocess_data('data/election_sentiment.csv', 'text', 'sentiment', 20, 5)
    anime_dict = preprocess_data('data/anime_reviews.csv', 'text', 'score', 50, 20)
   
def preprocess_data(file_path : str, text_var: str, class_var: str, lemm_length: int, unk_num: int):
    'Preprocesses a dataset generically, based on the passed in parameters'

    csv = pd.read_csv(file_path, usecols=[text_var, class_var], engine='python')  # keep only relevant columns in the dataset

    # 0. additional preprocessing for anime review dataset
    if (file_path == 'data/anime_reviews.csv'):
        csv = csv[csv[class_var] != (csv[class_var] < 3) | (csv[class_var] > 7)] # keeps only highly positive/negative reviews from anime dataset
        csv[class_var] = [1 if s > 7 else 0 for s in csv[class_var]] # turn positive sentiment into 1, negative sentiment into 0
        csv[text_var] = csv[text_var].apply(lambda x: ' '.join(x.split()[14:]))
    
    csv = csv[csv[class_var] != 'neutral'] # removes neutral rows if necessary
    csv[class_var] = [1 if s == 'positive' else 0 for s in csv[class_var]] # turn positive sentiment into 1, negative sentiment into 0


    # DENOISING
    # 1. noise removal (iterates through the dataframe column)
    def clean_noise(r):
        r = re.sub('<br\s*\/?>', ' ', r) # removes <br> tags
        r = re.sub('[^\w\s\d]|http\S+|<.*?>', ' ', r) # removes numbers, hyperlinks, and special characters
        r = re.sub('[\s\n]+', ' ', r.lower().strip()) # removes trailing whitespace and new lines
        return r

    csv[text_var] = csv[text_var].apply(lambda r: clean_noise(r)) # lambda instead of for loop for efficiency

    # 2. stop words & duplicate removal
    stop_word_list = set(nltk.corpus.stopwords.words('english'))
    def remove_stop_words(r):
        words = nltk.tokenize.word_tokenize(r)
        filtered_review = [word for word in words if word not in stop_word_list] # checks against nltk corpus
        return ' '.join(filtered_review)
    
    csv[text_var] = csv[text_var].apply(lambda r: remove_stop_words(r)) # lambda instead of for loop for efficiency
    

    # PREPROCESSING
    # randomly split examples into training and testing sets
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(csv[text_var], csv[class_var], test_size=0.3, random_state=42)
    vocabulary = {} 
    tkn_train_reviews = []
    tkn_test_reviews = []

    # 3. tokenization
    for review in train_reviews:
        tokens = nltk.word_tokenize(review)
        if len(review) > lemm_length:  # 4. lemmatize if review word length >lemm_length and truncate, pad if <lemm_length
            lemmatizer = nltk.WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        if len(tokens) > lemm_length:
            tokens = tokens[:lemm_length]  # truncation post-lemmatization (preserving significant features)
        else:
            tokens += ['<unk>'] * (lemm_length - len(tokens)) # padding
        tkn_train_reviews.append(tokens)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = 1  # count presence of word
            else:
                vocabulary[token] += 1

    for review in test_reviews:
        tokens = nltk.word_tokenize(review)
        if len(review) > lemm_length:  # 4. lemmatize if review word length >lemm_length and truncate, pad if <lemm_length
            lemmatizer = nltk.WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        if len(tokens) > lemm_length:
            tokens = tokens[:lemm_length]  # truncation post-lemmatization (preserving significant features)
        else:
            tokens += ['<unk>'] * (lemm_length - len(tokens)) # padding
        tkn_test_reviews.append(tokens)

    # convert rare words (<20 appearances) to <unk> (done separately on training and testing since had to split to compute vocabulary)
    to_pop = []
    for i, tokens in enumerate(tkn_train_reviews):
        for j, token in enumerate(tokens):
            if token in vocabulary and vocabulary[token] < unk_num:
                tkn_train_reviews[i][j] = '<unk>'
                to_pop.append(token)

    for i, tokens in enumerate(tkn_test_reviews):
        for j, token in enumerate(tokens):
            if token in vocabulary and vocabulary[token] < unk_num:
                tkn_test_reviews[i][j] = '<unk>'
                to_pop.append(token)
            elif token not in vocabulary:
                tkn_test_reviews[i][j] = '<unk>'

    for tkn in to_pop:
        if tkn in vocabulary:
            vocabulary.pop(tkn)

    print(tkn_train_reviews[:5])

    # 5. build a vocabulary with unique indexes for each word
    idx = 0
    for token in vocabulary:
        vocabulary[token] = idx
        idx += 1
    vocabulary['<unk>'] = len(vocabulary) # adding <UNK> to the vocabulary

    # 6. feature vectorization
    train_reviews = [[vocabulary.get(token, len(vocabulary)) for token in tokens] for tokens in tkn_train_reviews]
    test_reviews = [[vocabulary.get(token, len(vocabulary)) for token in tokens] for tokens in tkn_test_reviews]

    print('Preprocessing of data from the ' + file_path + ' filepath complete!')
    
    return dict(
        train_reviews          = np.array(train_reviews),
        test_reviews           = np.array(test_reviews),
        train_labels           = train_labels,
        test_labels            = test_labels,
        vocabulary             = vocabulary
    )

def load_data(datasource: str):
    'Returns the saved dict of word embeddings based on the provided filepath'
    if (datasource == 'movies'):
        return imdb_dict
    elif (datasource == 'airlines'):
        return airline_dict
    elif (datasource == 'elections'):
        return election_dict
    elif (datasource == 'anime'):
        return anime_dict

if __name__ == '__main__':
    preprocess()