from numpy import *


# this backprop implementation fits RBFN requirements, as its assumed that only 1 layer is to update
def back_propagation(A: array, W: array, Y: array):
    # A : activation matrix
    # W : weights matrix
    bias = random.random(1)
    epochs = 100
    learning_rate = 0.001

    for epoch in range(epochs):
        F = A.T.dot(W) + bias
        error = -(Y[epoch] - F).flatten()

        W = W - learning_rate * A * error
        bias = bias - learning_rate * error

    return W
