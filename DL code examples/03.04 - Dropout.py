import numpy as np
p = 0.5
W1, W2, W3, b1, b2, b3 = 1, 2, 3, 4, 5, 6 # weights to learn
def train_step(X): # X is the data
    # forward pass for a 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p # first dropout mask
    H1 *= U1
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p # / p # with the inverted dropout we don't have to scale after
    H2 *= U2
    out = np.dot(W3, H2) + b3

    # after this insert the backward pass

train_step(5)

def predict(X):
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p # at test time we have to scale
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * p
    out = np.dot(W3, H2) + b3