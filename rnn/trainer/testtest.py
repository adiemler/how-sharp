import argparse
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io

if __name__ == '__main__':

    args = parser.parse_args()

    DATA_PATH = 'data'
    OUTPUT_PATH = 'output'
    BATCH_SIZE = 256
    EPOCHS = 100
    NAME = '/rnn-sb'

    ''' Load data '''
    fs = file_io.FileIO(DATA_PATH + '/glove-100d-embeddings.pickle', 'rb')
    embeddings = pickle.load(fs)
    fs = file_io.FileIO(DATA_PATH + '/data.pickle', 'rb')
    xTrain, yTrain, xTest, yTest = pickle.load(fs)

    ''' Build model '''
    sentence = Input(shape=(1000,))  # sequence length from dataset
    embedded = Embedding(len(embeddings), 100, weights=[embeddings], input_length=1000, trainable=True)(sentence)
    lstm = Bidirectional(LSTM(100))(embedded)
    preds = Dense(50, activation='softmax')(lstm)
    #model = Model(sentence, preds)
    model = Model(sentence, preds)
    # same optimizer and loss as original paper
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ''' Train model 5 times '''
    trainSize = len(xTrain)  # 500
    testSize = len(xTest)  # trainSize // 5
    for i in range(3, 5):
        checkpoint = ModelCheckpoint('model-best.h5',
                                     monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True)
        history = model.fit(xTrain[:trainSize], yTrain[:trainSize],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=[xTest[:testSize], yTest[:testSize]],
                            callbacks=[checkpoint])
        model.save('model-final.h5')  # Save model locally

        # Copy best and final model to bucket
        with file_io.FileIO('model-best.h5', 'rb') as input_f:
            with file_io.FileIO(OUTPUT_PATH + NAME + '-best-%d.h5' % i, 'wb') as output_f:
                output_f.write(input_f.read())
        with file_io.FileIO('model-final.h5', 'rb') as input_f:
            with file_io.FileIO(OUTPUT_PATH + NAME + '-final-%d.h5' % i, 'wb') as output_f:
                output_f.write(input_f.read())

        # Save history for plotting later
        hist = history.history
        with file_io.FileIO(OUTPUT_PATH + NAME + '-history-%d.csv' % i, 'w') as f:
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
