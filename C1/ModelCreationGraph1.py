from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils as np_utils


parser = argparse.ArgumentParser()

#Adds arguments to command line so this file doesn't need constant editing, stores in constant variables
parser.add_argument('--output', help='Path to store output', required=True)
parser.add_argument('--epochs', help='Epochs', type=int, default=40)

args = parser.parse_args()

OUTPUT = args.output
EPOCHS = args.epochs

batch_sizes = [256, 5000]

#Load data (modeled after code from Keskar et al.)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#Convert data to desired format
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices - 10 nb_classes
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Build network (modeled after code from Keskar et al.)
img_size = (32, 32, 3)
model = Sequential()
model.add(Convolution2D(64, 5, 5, padding='same', input_shape=img_size))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

model.add(Convolution2D(64, 5, 5, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(384))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(192))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Set up model (modeled after code from Keskar et al.)
model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

#train models for both batch sizes
for BATCH_SIZE in batch_sizes:
	#train model 5 times
	for i in range(1, 6):

		#Train model (modeled after code from Keskar et al.)
		history = model.fit(X_train, Y_train,
					batch_size=BATCH_SIZE,
					nb_epoch=EPOCHS,
					validation_data=(X_test, Y_test),
					shuffle=True)

		# Save accuracies for plotting
		hist = history.history
		#print(hist.keys())
		with file_io.FileIO(OUTPUT + 'C1-accuracy-size-%i-num-%i.csv' % (BATCH_SIZE, i), 'w') as f: 
			f.write('Training,Testing\n')
			for j in range(EPOCHS):
				f.write('%f,%f\n' % (hist['acc'][j],
								   hist['val_acc'][j])) #Scale of 0-1, scale as needed when graphing
		
		#Save model for calculating other graphs
		model_json = model.to_json()
		with open("C1Model-size-%i-num-%i.json" % (BATCH_SIZE, i), "w") as json_file:
			json_file.write(model_json)
		#Save weights
		model.save_weights("C1Model-size-%i-num-%i-weights.h5" % (BATCH_SIZE, i))
		
		# Create new model by restarting session
		K.get_session().close()
		K.set_session(tf.Session())
		K.get_session().run(tf.global_variables_initializer())