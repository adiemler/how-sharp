import argparse
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        help='Path to training data',
                        required=True
                        )
    parser.add_argument('--output',
                        help='Path to output folder',
                        required=True
                        )
    parser.add_argument('--job-dir',
                        help='Path to job folder',
                        )
    parser.add_argument('--batch_size',
                        help='Batch size',
                        type=int,
                        default=256
                        )
    parser.add_argument('--epochs',
                        help='Epochs',
                        type=int,
                        default=40
                        )

    args = parser.parse_args()

    DATA_PATH = args.data
    OUTPUT_PATH = args.output
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NAME = '/rnn-sb'

    ''' Load pre-processed data '''
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
                  metrics=['sparse_categorical_accuracy'])

    ''' Train model 5 times '''
    trainSize = len(xTrain)  # 500
    testSize = len(xTest)  # trainSize // 5
    for i in range(1, 5):
        bestCheckpoint = ModelCheckpoint('model-best.h5',
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True)
        epochCheckpoint = ModelCheckpoint('model-epoch-{epoch:d}.h5',
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
        model.save('model-final.h5')  # Save model locally

        # Copy models to bucket
        for e in range(EPOCHS):
            with file_io.FileIO('model-%d-epoch-%d.h5' % (i, e + 1), 'rb') as input_f:
                with file_io.FileIO(OUTPUT_PATH + NAME + '-epoch-%d.h5' % i, 'wb') as output_f:
                    output_f.write(input_f.read())
        with file_io.FileIO('model-best.h5', 'rb') as input_f:
            with file_io.FileIO(OUTPUT_PATH + NAME + '-best-%d.h5' % i, 'wb') as output_f:
                output_f.write(input_f.read())
        with file_io.FileIO('model-final.h5', 'rb') as input_f:
            with file_io.FileIO(OUTPUT_PATH + NAME + '-final-%d.h5' % i, 'wb') as output_f:
                output_f.write(input_f.read())

        # Save history for plotting later
        hist = history.history
        print(hist.keys())
        with file_io.FileIO(OUTPUT_PATH + NAME + '-history-%d.csv' % i, 'w') as f:
            f.write('loss,acc,val_loss,val_acc\n')
            for k in range(EPOCHS):
                f.write('%f,%f,%f,%f\n' % (hist['loss'][k],
                                           hist['sparse_categorical_accuracy'][k],
                                           hist['val_loss'][k],
                                           hist['val_sparse_categorical_accuracy'][k]))

        # Restart session to re-initialize variables
        K.get_session().close()
        K.set_session(tf.Session())
        K.get_session().run(tf.global_variables_initializer())
