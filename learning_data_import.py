import numpy as np
from numpy import genfromtxt
from config import ROOT_DIR
import os


def custom_function1(n: int):
    X = np.mgrid[0:4:complex(0, n)].reshape(n, 1)
    Y = np.array([np.sin(x) ** 3 * np.exp(x / 10) for x in X])
    return X, Y


def custom_function2(n: int):
    X = np.mgrid[0:4:complex(0, n)].reshape(n, 1)
    Y = np.array([np.sin(x) ** 3 * np.exp(x / 10) * (np.random.rand(1) / 10 + 1) for x in X])
    return X, Y


def school_data_set():
    path = os.path.join(ROOT_DIR, "data", "approximation_test.txt")
    data = genfromtxt(path, delimiter=' ')
    X = data[:, 0]
    Y = data[:, 1]
    arg_sort = X.argsort()
    Y = Y[arg_sort].reshape(len(Y), 1)
    X = X[arg_sort].reshape(len(X), 1)
    return X, Y


def classification_learning_set():
    path = os.path.join(ROOT_DIR, "data", "classification_test.txt")
    data = genfromtxt(path, delimiter=' ')
    X = np.array(data[:, range(4)])
    X = X.reshape(len(X), 4)
    Y = data[:, 4]
    Y = Y.reshape(len(Y), 1)
    return X, Y


def classification_test_set():
    path = os.path.join(ROOT_DIR, "data", "classification_train.txt")
    data = genfromtxt(path, delimiter=' ')
    X = np.array(data[:, range(4)])
    X = X.reshape(len(X), 4)
    Y = data[:, 4]
    Y = Y.reshape(len(Y), 1)
    return X, Y
