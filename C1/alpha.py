from scipy.optimize import minimize
from tensorflow.keras import backend as K, losses as losses
from tensorflow.keras.datasets import imdb, cifar10
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from time import time
import numpy as np
import sys

# batch size doesn't affect loss value, just batch size keras uses when 
# calculating it. Set it as big as possible for faster evaluation
BATCH_SIZE = 2048
SB_PATH = sys.argv[1]
LB_PATH = sys.argv[2]
LOSS = losses.categorical_crossentropy

''' Load data '''
print('Loading data...')
start = time()
(X, Y), (XTest, YTest) = cifar10.load_data()
#Convert data to desired format
X = X.astype('float32')
XTest = XTest.astype('float32')
X /= 255
XTest /= 255
# convert class vectors to binary class matrices - 10 nb_classes
Y = to_categorical(Y, 10)
YTest = to_categorical(YTest, 10)

with open(SB_PATH, 'r') as f:
    loadedJson = f.read()
model = model_from_json(loadedJson)
sbWeights = []
for i in range(len(model.layers)):
    for w in model.layers[i].get_weights():
        flat = np.ndarray.flatten(w)
        for f in flat:
            sbWeights.append(f)

with open(LB_PATH, 'r') as f:
    loadedJson = f.read()
model = model_from_json(loadedJson)
lbWeights = []
for i in range(len(model.layers)):
    for w in model.layers[i].get_weights():
        flat = np.ndarray.flatten(w)
        for f in flat:
            lbWeights.append(f)
print('Loaded in %.2fs' % (time() - start))

def f(alpha):
    weights = [0 for i in range(len(sbWeights))]
    for i in range(len(weights)):
        weights[i] = alpha * lbWeights[i] + (1 - alpha) * sbWeights[i]
        
    start = 0
    for i in range(len(model.layers)):
        layerWeights = []
        for k in range(len(model.layers[i].get_weights())):
            shape = model.layers[i].get_weights()[k].shape
            count = np.prod(shape)
            group = weights[start:start+count]
            start += count
    
            group = np.reshape(group, shape)
            layerWeights.append(group)
        model.layers[i].set_weights(layerWeights)
            
    # return loss and accuracy metrics
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    loss, acc = model.evaluate(X, Y, batch_size=BATCH_SIZE)
    valLoss, valAcc = model.evaluate(XTest, YTest, batch_size=BATCH_SIZE)
    return (loss, acc, valLoss, valAcc)

print('Calculating...')
start = time()

print('loss,acc')
for alpha in np.linspace(-1, 2, 25):
    print('%f,%f,%f,%f' % f(alpha))

print('Calculated in %.2fs' % (time() - start))
