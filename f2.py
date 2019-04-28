from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils as np_utils

num_classes = 10
epochs = 12
a = []

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255

number_of_classes = 10

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)



parser = argparse.ArgumentParser()

#Adds arguments to command line so this file doesn't need constant editing, stores in constant variables
parser.add_argument('--output', help='Path to store output', required=True)
parser.add_argument('--epochs', help='Epochs', type=int, default=40)

args = parser.parse_args()

OUTPUT = args.output
EPOCHS = args.epochs

batch_sizes = [256, 5000]


#Build the network
model = Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_dim=784))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


for BATCH_SIZE in batch_sizes:
	#train model 5 times
	for i in range(1,5):

		#Train model (modeled after code from Keskar et al.)
		history = model.fit(x_train, y_train,
					batch_size=BATCH_SIZE,
					nb_epoch=EPOCHS,
					validation_data=(x_test, y_test),
					shuffle=True)
		score = model.evaluate(x_test, y_test)
		print ('Test Accuracy: ', score)
		a.append(score)
		# Save accuracies for plotting
		hist = history.history
		print(hist.keys())
		with file_io.FileIO(OUTPUT + 'F1-accuracy-size-%i-num-%i.csv' % (BATCH_SIZE, i), 'w') as f: 
			f.write('Training,Testing\n')
			for j in range(EPOCHS):
				f.write('%f,%f\n' % (hist['acc'][j],
								   hist['val_acc'][j])) #Scale of 0-1, scale as needed when graphing
		
		#Save model for calculating other graphs
		model_json = model.to_json()
		with open("F1Model-size-%i-num-%i.json" % (BATCH_SIZE, i), "w") as json_file:
			json_file.write(model_json)
		#Save weights
		model.save_weights("F1Model-size-%i-num-%i-weights.h5" % (BATCH_SIZE, i))
		
		# Create new model by restarting session
		K.get_session().close()
		K.set_session(tf.Session())
		K.get_session().run(tf.global_variables_initializer())

print(a)

