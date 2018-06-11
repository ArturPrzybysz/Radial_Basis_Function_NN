from numpy import *


# this backprop implementation fits RBFN requirements, as its assumed that only 1 layer is to update
def back_propagation(A: array, W: array, Y: array):
    # A : activation matrix
    # W : weights matrix

    bias = 0
    epochs = 1000
    initial_learning_rate = 0.1
    final_learning_rate = 0.001
    for epoch in range(epochs):
        learning_rate = initial_learning_rate * pow(final_learning_rate / initial_learning_rate, epoch / epochs)
        for i in range(len(A)):
            a = A[i]
            F = a.T.dot(W) + bias
            error = -(Y[i] - F).flatten()
            a = a.reshape(W.shape[0], 1)
            W = W - learning_rate * error * a
            bias = bias - learning_rate * error
    return W, bias
