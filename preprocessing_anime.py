import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data():
    '''
    Method that was used to preprocess the data in the imdb_dataset.csv file.
    '''

    csv_file_path = f'data/anime_reviews.csv'

    csv = pd.read_csv(csv_file_path, usecols=['text', 'score'], engine='python')  # keep only relevant columns in the dataset
    print(csv.head())

    # remove neutral rows
    csv = csv[csv.score != (csv.score < 3) | (csv.score > 7)]
    # turn positive sentiment into 1, negative sentiment into 0
    csv.score = [1 if s > 7 else 0 for s in csv.score]

    # DENOISING
    # 0. removing the repeated text from the start of each review
    csv['text'] = csv['text'].apply(lambda x: ' '.join(x.split()[14:]))

    # 1. noise removal (iterates through the dataframe column)
    def clean_noise(r):
        r = re.sub('<br\s*\/?>', ' ', r) # removes <br> tags
        r = re.sub('[^\w\s\d]|http\S+|<.*?>', ' ', r) # removes numbers, hyperlinks, and special characters
        r = re.sub('[\s\n]+', ' ', r.lower().strip()) # removes trailing whitespace and new lines
        return r

    csv['text'] = csv['text'].apply(lambda r: clean_noise(r)) # lambda instead of for loop for efficiency

    # 2. stop words & duplicate removal
    stop_word_list = set(nltk.corpus.stopwords.words('english'))
    def remove_stop_words(r):
        words = nltk.tokenize.word_tokenize(r)
        filtered_review = [word for word in words if word not in stop_word_list] # checks against nltk corpus
        return ' '.join(filtered_review)
    
    csv['text'] = csv['text'].apply(lambda r: remove_stop_words(r)) # lambda instead of for loop for efficiency
    print(csv.head())
    ''
    # PREPROCESSING
    # randomly split examples into training and testing sets
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(csv['text'], csv['score'], test_size=0.3, random_state=42)
    vocabulary = {} 
    tkn_train_reviews = []
    tkn_test_reviews = []

    # 3. tokenization
    for review in train_reviews:
        tokens = nltk.word_tokenize(review)
        if len(review) > 50:  # 4. lemmatize if review word length >25 and truncatem, pad if <25
            lemmatizer = nltk.WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        if len(tokens) > 50:
            tokens = tokens[:50]  # truncation post-lemmatization (preserving significant features)
        else:
            tokens += ['<unk>'] * (50 - len(tokens)) # padding
        tkn_train_reviews.append(tokens)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = 1  # count presence of word
            else:
                vocabulary[token] += 1

    for review in test_reviews:
        tokens = nltk.word_tokenize(review)
        if len(review) > 50:  # 4. lemmatize if review word length > 25
            lemmatizer = nltk.WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        if len(tokens) > 50:
            tokens = tokens[:50]  # truncation post-lemmatization (preserving significant features)
        else:
            tokens += ['<unk>'] * (50 - len(tokens)) # padding
        tkn_test_reviews.append(tokens)


    # print(train_reviews[:5])
    # print(tkn_train_reviews[:5])

    # convert rare words (<20 appearances) to <unk> (done separately on training and testing since had to split to compute vocabulary)
    to_pop = []
    for i, tokens in enumerate(tkn_train_reviews):
        for j, token in enumerate(tokens):
            if token in vocabulary and vocabulary[token] < 30:
                tkn_train_reviews[i][j] = '<unk>'
                to_pop.append(token)

    for i, tokens in enumerate(tkn_test_reviews):
        for j, token in enumerate(tokens):
            if token in vocabulary and vocabulary[token] < 30:
                tkn_test_reviews[i][j] = '<unk>'
                to_pop.append(token)
            elif token not in vocabulary:
                tkn_test_reviews[i][j] = '<unk>'

    for tkn in to_pop:
        if tkn in vocabulary:
            vocabulary.pop(tkn)

    # 5. build a vocabulary with unique indexes for each word
    idx = 0
    for token in vocabulary:
        vocabulary[token] = idx
        idx += 1
    vocabulary['<unk>'] = len(vocabulary) # adding <UNK> to the vocabulary

    # 6. feature vectorization
    train_reviews = [[vocabulary.get(token, len(vocabulary)) for token in tokens] for tokens in tkn_train_reviews]
    test_reviews = [[vocabulary.get(token, len(vocabulary)) for token in tokens] for tokens in tkn_test_reviews]
    # print('current vectorized reviews are ' + str(train_reviews))
    

    return dict(
        train_reviews          = np.array(train_reviews),
        test_reviews           = np.array(test_reviews),
        train_labels           = train_labels,
        test_labels            = test_labels,
        vocabulary             = vocabulary
    )


if __name__ == '__main__':
    load_data()