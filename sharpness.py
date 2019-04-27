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
BATCH_SIZE = 4096
INPUT_DIM = 300
MODEL_PATH = sys.argv[1]
LOSS = losses.binary_crossentropy

''' Load data '''
print('Loading data...')
start = time()
(X, Y), (_, _) = imdb.load_data()
X = pad_sequences(X, maxlen=300)
X = X.reshape(len(X), 300, 1)
model = load_model(MODEL_PATH)
weights = []
for i in range(len(model.layers)):
    for w in model.layers[i].get_weights():
        for sub in w:
            try:
                for i in range(len(sub)):
                    weights.append(sub[i])
            except:
                weights.append(sub)
n = len(weights)
print('Loaded in %.2fs' % (time() - start))

def f(z, A):
    '''
    The loss function.
    Based on constrained variable z and dimension reduction matrix A.
    '''
    adjustment = np.matmul(A, z)
    for i in range(len(weights)):
        weights[i] += adjustment[i]
        
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
            
    metrics = model.evaluate(X, Y, batch_size=BATCH_SIZE)
    # we're passing it to a minimizer, so it needs to be negative
    # if we want to maximize
    return -1 * metrics[0]

def boundedMax(eps, A):
    p = len(A[0])
    Ainv = np.linalg.pinv(A)
    AW = np.matmul(Ainv, weights)
    bounds = []
    # {z in Rp: -eps(|(Ainv * x)i| + 1) <= zi <= eps(|Ainv * x)i| + 1)}
    for i in range(p):
        lower = -eps * (abs(AW[i]) + 1)
        upper = eps * (abs(AW[i]) + 1)
        bounds.append((lower, upper))

    boundedMax = minimize(f,  # objective function
                          np.zeros(p),  # initial guess
                          args=(A),  # additional arguments
                          method='L-BFGS-B',  # method for optimization
                          bounds=bounds,  # limits for each dim of input
                          options={maxiter: 10})  # max iterations, same as paper
    # we flipped the sign in f, so we'll flip it back here for the true value
    return -1 * boundedMax

# Two different sharpness constants
eps = [0.001, 0.0005]
# Full space
Afull = np.identity(n)
# Random subspace
cols = np.random.choice(n, size=100, replace=False)
Arand = Afull[:,cols]
As = [Afull, Arand]

# final error of model is independent of eps and A
print('test')
print(model.evaluate(X, Y, batch_size=BATCH_SIZE)[0])
error = f(np.zeros(n), np.identity(n))
print(error * -1)

print('Calculating sharpness...')
start = time()
for e in eps:
    for A in As:
        # phi(eps, A) = 100 * (max for z in C (f(x + Az)) - f(x)) / (1 + f(x))
        bm = boundedMax(e, A)
        sharpness = bm - error
        sharpness /= 1 + error
        sharpness *= 100
        print('Eps=%f, p=%d, sharpness=%f' % (e, len(A[0]), sharpness))
print('Calculated in %.2fs' % (time() - start))
