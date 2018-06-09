import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from back_propagation import back_propagation


class rbfn:
    def __init__(self, input_size, output_size, centers_count,
                 center_assignation: str = "random",
                 weight_update: str = "pinv"):
        self.input_size = input_size
        self.output_size = output_size
        self.centers_count = centers_count

        self.centers = np.zeros(self.centers_count)
        self.weights = np.random.rand(centers_count, output_size) * 2 - 1

        self.center_assignation = center_assignation
        self.weight_update = weight_update

        self.beta = - 10

    def _rb_function(self, centers, x):
        return np.exp(self.beta * np.linalg.norm(centers - x) ** 2)

    def _calculate_activations(self, X):
        A = np.zeros((len(X), self.centers_count))

        for c_i, c in enumerate(self.centers):
            for x_i, x in enumerate(X):
                A[x_i, c_i] = self._rb_function(c, x)

        return A

    def _assign_centers(self, X):
        if self.center_assignation == "random":
            rnd_idx = np.random.permutation(X.shape[0])[:self.centers_count]
            self.centers = [X[i, :] for i in rnd_idx]
            return

        if self.center_assignation == "kmeans":
            pass
            # TODO implement k means center assignation

    def _update_weights(self, A, Y):
        if self.weight_update == "pinv":
            self.weights = np.dot(np.linalg.pinv(A), Y)
        elif self.weight_update == "backprop":
            self.weights = back_propagation(A, self.weights, Y)

    def train(self, X, Y):
        self._assign_centers(X)
        A = self._calculate_activations(X)
        self._update_weights(A, Y)

    def test(self, X):
        A = self._calculate_activations(X)
        return np.dot(A, self.weights)
