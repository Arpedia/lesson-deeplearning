import numpy as np

def step(x):
    boolx = x.any() > 0
    return boolx.astype(np.int)
    # return np.where(np.array(x > 0), np.ones_like(x), np.zeros_like(x));

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    C = np.max(x, axis = 1)
    exp_x = np.exp(x - C[:, np.newaxis] * np.ones_like(x))
    sum_exp_x = np.sum(exp_x, axis = 1)
    y = exp_x / sum_exp_x[:, np.newaxis]
    return y
