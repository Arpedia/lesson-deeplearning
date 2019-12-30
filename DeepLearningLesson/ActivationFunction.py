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
    C = np.max(x)
    exp_x = np.exp(x - C)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp
