import sys, os
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNetwork import TwoLayerNetwork
from matplotlib import pyplot
from Optimazation import StochasticGradientOptimize, MomentumOptimize

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)

train_loss_list = []

# Hyper Parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.4

network = TwoLayerNetwork(x_train.shape[1], 50, 10)

iter_per_epoch = max(train_size / batch_size, 1)

#optimizer = StochasticGradientOptimize(learning_rate)
optimizer = MomentumOptimize(learning_rate)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #grads = network.gradient(x_batch, t_batch)
    grads = network.numerical_gradient(x_batch, t_batch)
    network.params = optimizer.update(network.params, grads)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(train_acc * 100, test_acc * 100)

print(network.accuracy(x_test, t_test) * 100)

pyplot.plot(train_loss_list)
pyplot.show()