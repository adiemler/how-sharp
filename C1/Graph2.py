#Python script that loads two models from files, and creates a csv file with the data for the 
#second graph in the paper by Keskar et. al
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.python.lib.io import file_io
import tensorflow.keras.models
import numpy
import argparse
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils as np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_value
import tensorflow as tf

#Adds arguments to command line so this file doesn't need constant editing, stores in constant variables
parser = argparse.ArgumentParser()

parser.add_argument('--output', help='Path to store output', required=True)
parser.add_argument('--number', help='Network number to perform the operation on',type = int, required=True)
#parser.add_argument('--augmented', help='Performing on augmented data? Graph 6', type=bool, default = false)

args = parser.parse_args()

OUTPUT = args.output
NUMBER = args.number
#AUGMENTED = args.augmented

#Load Data (from ModelCreationGraph1.py)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#Convert data to desired format
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices - 10 nb_classes
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

'''
#Augment data if necessary
if AUGMENTED:
	datagen = ImageDataGenerator(
				rotation_range=10,
				horizontal_flip=True,
				width_shift_range=0.2)
	datagen.fit(X_train)
	datagen.fit(X_test)
'''
	

#Calculate for specified instance 
#Load SB model
#if AUGMENTED:
#	json_file = open(INPUT + "C1AugModel-size-256-num-%i.json" % NUMBER, 'r')
#else:
json_file = open("C1Model-size-256-num-%i.json" % NUMBER, 'r')

model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
#if AUGMENTED:
#	model.load_weights(INPUT + "C1AugModel-size-256-num-%i-weights.h5" % NUMBER)
#else:
model.load_weights("C1Model-size-256-num-%i-weights.h5" % NUMBER)
model.compile(loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])

SB_weights = []
#Get weights from small batch model
for layer in range(len(model.layers)):
	SB_weights.append(model.layers[layer].trainable_weights)
#SB_weights = model.trainable_weights

#Load LB model
#if AUGMENTED:
#	json_file = open("C1AugModel-size-5000-num-%i.json" % NUMBER, 'r')
#else:
json_file = open("C1Model-size-5000-num-%i.json" % NUMBER, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
#if AUGMENTED:
#	model.load_weights("C1AugModel-size-5000-num-%i-weights.h5" % NUMBER)
#else:
model.load_weights("C1Model-size-5000-num-%i-weights.h5" % NUMBER)
model.compile(loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])

LB_weights = []
#Get weights from large batch model
for layer in range(len(model.layers)):
	LB_weights.append(model.layers[layer].trainable_weights)
#LB_weights = model.trainable_weights

#establish range of alpha
alpha_range = numpy.linspace(-1, 2, 25) #Splits range of -1 and 2 into even slices

#open csv file and print header
#if AUGMENTED:
#	FILE_NAME = 'C1Aug-CrossEntropy-Accuracy-Alpha-%i.csv' % NUMBER
#else:
FILE_NAME = 'C1-CrossEntropy-Accuracy-Alpha-%i.csv' % NUMBER
with file_io.FileIO(OUTPUT + FILE_NAME, 'w') as f: 
	f.write('Alpha,TrainCrossEntropy,TrainAccuracy,TestCrossEntropy,TestAccuracy\n')
		
	
	#Calculate cross entropy and accuracy for these two models (modeled after code from Keskar et. al)
	for alpha in alpha_range:
	
		#Resetting model
		print('Resetting model...\n')
		json_file = open("C1Model-size-5000-num-%i.json" % NUMBER, 'r')
		model_json = json_file.read()
		json_file.close()
		model = model_from_json(model_json)
		model.load_weights("C1Model-size-5000-num-%i-weights.h5" % NUMBER)
		model.compile(loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])
		
		print(alpha)
		
		for layer in range(len(model.layers)):
			for p in SB_weights[layer]:
				term1 = tf.multiply(LB_weights[layer][p], alpha)
				term2 = tf.multiply(SB_weights[layer][p], (1-alpha))
				combined[p] = tf.add(term1, term2)
				#LB_weights[p]*alpha + SB_weights[p]*(1-alpha)
			
			model.layers[layer].set_weights(combined)
			#tf.assign(model.layer[layer].settrainable_weights[p], combined)
			print('\n')
			print(combined)
			print('\n')
				#set_value(model.trainable_weights[p], new_weights[p]);
		
		print(' Assigned, evaluating...')
		train_xent, train_acc = model.evaluate(X_train, Y_train,
											   batch_size=5000, verbose=0)
		print('...')
			
		test_xent, test_acc = model.evaluate(X_test, Y_test,
											 batch_size=5000, verbose=0)
		print('\n')
		#print data
		f.write('%f,%f,%f,%f,%f\n' % (alpha,
										train_xent,
										train_acc,
										test_xent,
										test_acc))
		