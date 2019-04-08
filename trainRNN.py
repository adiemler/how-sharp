import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy.random import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model

NAME = 'Models/RNN/rnn-sb'
BATCH_SIZE = 256
EPOCHS = 100 # 3

print('Loading data...')
with open('Data/Authors/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
embeddings = np.load('Data/Authors/glove-100d-embeddings.npy')
data = np.load('Data/Authors/data.npz')
xTrain = data['xTrain']
xTest = data['xTest']
yTrain = data['yTrain']
yTest = data['yTest']

print('Building model...')
sentence = Input(shape=(1000,)) # sequence length from dataset
embedded = Embedding(len(embeddings), 100, weights=[embeddings], input_length=1000, trainable=True)(sentence)
lstm = Bidirectional(LSTM(100))(embedded)
preds = Dense(50, activation='softmax')(lstm)
model = Model(sentence, preds)
# same optimizer and loss as original paper
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training...')
trainSize = len(xTrain) # 500
testSize = len(xTest) # trainSize // 5
for i in range(5):

    checkpoint = ModelCheckpoint(NAME + '-best-%d.h5' % i, 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True)
    history = model.fit(xTrain[:trainSize], yTrain[:trainSize], 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        validation_data=[xTest[:testSize], yTest[:testSize]],
        callbacks=[checkpoint])
    model.save(NAME + '-final-%d.h5' % i)

    # Save history for plotting later
    hist = history.history
    with open(NAME + '-history-%d.csv' % i, 'w') as f:
        f.write('loss,acc,val_loss,val_acc\n')
        for k in range(EPOCHS):
            f.write('%f,%f,%f,%f\n' % (hist['loss'][k], 
                                       hist['acc'][k], 
                                       hist['val_loss'][k], 
                                       hist['val_acc'][k]))

    # Restart session to re-initialize variables
    K.get_session().close()
    K.set_session(tf.Session())
    K.get_session().run(tf.global_variables_initializer())