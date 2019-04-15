import pickle
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Bidirectional, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time

OUTPUT_PATH = 'output/' + sys.argv[1]
BATCH_SIZE = 2048
EPOCHS = 100
NAME = '/rnn-lb'

''' Load data '''
start = time()
(xTrain, yTrain), (xTest, yTest) = imdb.load_data()
xTrain = pad_sequences(xTrain, maxlen=1000)
xTest = pad_sequences(xTest, maxlen=1000)
xTrain = xTrain.reshape(len(xTrain), 1000, 1)
xTest = xTest.reshape(len(xTest), 1000, 1)
print('Loaded data in %.2fs' % (time() - start))

''' Build model '''
review = Input(shape=(1000,1))
lstm = Bidirectional(LSTM(100))(review)
preds = Dense(1, activation='softmax')(lstm)
model = Model(review, preds)
# same optimizer and loss as original paper
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
print('Built model in %.2fs' % (time() - start))

''' Train model '''
trainSize = len(xTrain)  # set to 500 for fast debugging
testSize = len(xTest)  # set to 50 for fast debugging
bestCheckpoint = ModelCheckpoint(OUTPUT_PATH + NAME + '-best.h5',
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True)
epochCheckpoint = ModelCheckpoint(OUTPUT_PATH + NAME + '-epoch-{epoch:d}.h5',
                                  monitor='val_loss',
                                  verbose=0,
                                  save_best_only=False,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)
history = model.fit(xTrain[:trainSize], yTrain[:trainSize],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=[xTest[:testSize], yTest[:testSize]],
                    callbacks=[bestCheckpoint, epochCheckpoint])
model.save(OUTPUT_PATH + NAME + '-final.h5')

# Save history for plotting later
hist = history.history
print(hist.keys())
with open(OUTPUT_PATH + NAME + '-history.csv', 'w') as f:
    f.write('loss,acc,val_loss,val_acc\n')
    for k in range(EPOCHS):
        f.write('%f,%f,%f,%f\n' % (hist['loss'][k],
                                   hist['acc'][k],
                                   hist['val_loss'][k],
                                   hist['val_acc'][k]))
