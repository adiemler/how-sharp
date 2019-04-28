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
MODEL_PATH = sys.argv[1]
LOSS = losses.categorical_crossentropy

''' Load data '''
print('Loading data...')
start = time()
(X, Y), (XTest, YTest) = imdb.load_data()
X = pad_sequences(X, maxlen=300)
XTest = pad_sequences(XTest, maxlen=300)
X = X.reshape(len(X), 300, 1)
XTest = XTest.reshape(len(XTest), 300, 1)

model = load_model(MODEL_PATH)
weights = []
for i in range(len(model.layers)):
    for w in model.layers[i].get_weights():
        flat = np.ndarray.flatten(w)
        for f in flat:
            weights.append(f)

n = len(weights)
print('Loaded in %.2fs' % (time() - start))

evalCount = 0
result = 0

def f(z, A=None):
    global evalCount
    global result
    print(evalCount)
    if evalCount > 30:
        evalCount = 0
        raise Exception
    evalCount += 1
    '''
    The loss function.
    Based on constrained variable z and dimension reduction matrix A.
    '''
    if A is None:
        adjustment = z
    else:
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
    result = metrics[0]
    return -1 * metrics[0]

def boundedMax(eps, A=None):
    if A is None:
        p = n
        bounds = []
        for i in range(p):
            lower = -eps * (abs(weights[i]) + 1)
            upper = eps * (abs(weights[i]) + 1)
            bounds.append((lower, upper))
    else:
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
                          np.random.rand(p) - 0.5,  # initial guess
                          args=(A),  # additional arguments
                          method='L-BFGS-B',  # method for optimization
                          bounds=bounds)  # limits for each dim of input
    # we flipped the sign in f, so we'll flip it back here for the true value
    return -1 * boundedMax

# Two different sharpness constants
eps = [0.001, 0.0005]

# Random subspace
cols = np.random.choice(n, size=100, replace=False)
A = np.zeros((n, 100))
for i in range(100):
    A[cols[i]][i] = 1

# final error of model is independent of eps and A
error = -1 * f(np.zeros(n))

print('Calculating sharpness...')
start = time()
for e in eps:
    # Don't calculate identity because that takes a 
    # --- .fuck. ---
    # ton of memory
    try:
        bm = boundedMax(e)
    except:
        bm = result
    sharpness = 100 * (bm - error) / (1 + error)
    print('Eps=%f, p=%d, sharpness=%f' % (e, n, sharpness))

    bm = boundedMax(e, A)
    bm = result
    sharpness = 100 * (bm - error) / (1 + error)
    print('Eps=%f, p=%d, sharpness=%f' % (e, 100, sharpness))
print('Calculated in %.2fs' % (time() - start))
