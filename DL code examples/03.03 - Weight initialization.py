import numpy as np
def weight_initialization_tanh():
    dims = [4096] * 7
    hs = []
    x = np.random.randn(16, dims[0])
    for Din, Dout in zip(dims[:-1], dims[1:]):
        W = np.random.randn(Din, Dout) / np.sqrt(Din)
        x = np.tanh(x.dot(W))
        hs.append(x)

    print(hs)

def weight_initialization_ReLU():
    dims = [4096] * 7
    hs = []
    x = np.random.randn(16, dims[0])
    for Din, Dout in zip(dims[:-1], dims[1:]):
        W = np.random.randn(Din, Dout) * np.sqrt(2/Din)
        x = np.maximum(0, x.dot(W))
        hs.append(x)

    print(hs)