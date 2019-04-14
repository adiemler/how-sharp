import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from time import time

OUTPUT_PATH = 'output/'
BATCH_SIZE = 256
EPOCHS = 2
NAME = '/rnn-sb'

''' Load data '''
start = time()
(xTrain, yTrain), (xTest, Test) = imdb.load_data()
print('Loaded data in %.2fs' % (time() - start))

''' Build model '''
start = time()
review = Input(shape=(None, 1))
lstm = Bidirectional(LSTM(100))(review)
preds = Dense(50, activation='softmax')(lstm)
model = Model(review, preds)
# same optimizer and loss as original paper
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
print('Built model in %.2fs' % (time() - start))

''' Train model 5 times '''
trainSize = len(xTrain)  # 500
testSize = len(xTest)  # trainSize // 5
for i in range(2):
    bestCheckpoint = ModelCheckpoint(OUTPUT_PATH + str(i) + NAME + '-best.h5',
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True)
    epochCheckpoint = ModelCheckpoint(OUTPUT_PATH + str(i) + NAME + '-epoch-{epoch:d}.h5',
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
    model.save(OUTPUT_PATH + str(i) + NAME + '-final.h5')

    # Save history for plotting later
    hist = history.history
    print(hist.keys())
    with file_io.FileIO(OUTPUT_PATH + str(i) + NAME + '-history-%d.csv' % i, 'w') as f:
        f.write('loss,acc,val_loss,val_acc\n')
        for k in range(EPOCHS):
            f.write('%f,%f,%f,%f\n' % (hist['loss'][k],
                                       hist['binary_accuracy'][k],
                                       hist['val_loss'][k],
                                       hist['val_binary_accuracy'][k]))

    # Restart session to re-initialize variables
    K.get_session().close()
    K.set_session(tf.Session())
    K.get_session().run(tf.global_variables_initializer())
