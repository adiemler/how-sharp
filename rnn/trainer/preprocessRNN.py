import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

PATH = '../data/'
SPLIT = 45000 # 53678 examples into 45000 train and 8678 test

def load():
    global PATH
    print('Loading data...')
    x = []
    y = []
    with open(PATH + 'data-train.csv', 'r') as f:
        f.readline() # throw away header row
        for line in f:
            text, author = line.split(',')
            x.append(text)
            '''
            # Don't need this if using sparse_categorical_crossentropy loss with tensorflow
            # One-hot-ify authors
            vec = np.zeros(50) # 50 authors
            vec[int(author) - 1] = 1
            y.append(vec)
            '''
            y.append(int(author) - 1)
    return (x, y)

def tokenize(x):
    '''
    Tokenize sentences into integers
    https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
    '''
    global PATH
    print('Tokenizing data...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    with open(PATH + 'tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer

def embed(tokenizer):
    '''
    Create matrix of embeddings ordered by indexes used in tokenizer
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    '''
    global PATH
    print('Loading embeddings...')
    vocabSize = len(tokenizer.word_index) + 1
    embeddings = np.zeros((vocabSize, 100))
    with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word, *values = line.split()
            # Only get words actually in our text
            if word in tokenizer.word_index:
                index = tokenizer.word_index[word]
                embeddings[index] = np.array(values, dtype='float32')
    np.save(PATH + 'glove-100d-embeddings.npy', embeddings)

x, y = load()
tokenizer = tokenize(x)
embed(tokenizer)

print('Splitting and saving data...')
encodedX = tokenizer.texts_to_sequences(x)
xTrain = encodedX[:SPLIT]
yTrain = y[:SPLIT]
xTest = encodedX[SPLIT:]
yTest = y[SPLIT:]

# Numpy-ify data
xTrain = np.array(xTrain)
xTest = np.array(xTest)
yTrain = np.array(yTrain)
yTest = np.array(yTest)

np.savez(PATH + 'data.npz', xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)