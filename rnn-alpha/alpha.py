from scipy.optimize import minimize
from tensorflow.keras import backend as K, losses as losses
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time
import numpy as np
import sys

# batch size doesn't affect loss value, just batch size keras uses when 
# calculating it. Set it as big as possible for faster evaluation
BATCH_SIZE = 2048
SB_PATH = sys.argv[1]
LB_PATH = sys.argv[2]
LOSS = losses.binary_crossentropy

''' Load data '''
print('Loading data...')
start = time()
(X, Y), (XTest, YTest) = imdb.load_data()
X = pad_sequences(X, maxlen=300)
XTest = pad_sequences(XTest, maxlen=300)
X = X.reshape(len(X), 300, 1)
XTest = XTest.reshape(len(XTest), 300, 1)
model = load_model(SB_PATH)
sbWeights = []
for i in range(len(model.layers)):
    for w in model.layers[i].get_weights():
        for sub in w:
            try:
                for i in range(len(sub)):
                    sbWeights.append(sub[i])
            except:
                sbWeights.append(sub)

model = load_model(LB_PATH)
lbWeights = []
for i in range(len(model.layers)):
    for w in model.layers[i].get_weights():
        for sub in w:
            try:
                for i in range(len(sub)):
                    lbWeights.append(sub[i])
            except:
                lbWeights.append(sub)
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
    loss, acc = model.evaluate(X, Y, batch_size=BATCH_SIZE)
    valLoss, valAcc = model.evaluate(XTest, YTest, batch_size=BATCH_SIZE)
    return (loss, acc, valLoss, valAcc)

print('Calculating...')
start = time()

print('loss,acc')
for alpha in np.linspace(-1, 2, 25):
    print('%f,%f,%f,%f' % f(alpha))

print('Calculated in %.2fs' % (time() - start))
