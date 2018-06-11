import numpy as np
from scipy.cluster.vq import kmeans
from back_propagation import back_propagation
from scipy.spatial.distance import euclidean


class rbfn:
    def __init__(self, input_size, centers_count,
                 center_assignation: str = "random",
                 weight_update: str = "pinv"):
        self.input_size = input_size
        self.centers_count = centers_count

        self.centers = np.zeros(self.centers_count)
        self.weights = np.random.rand(centers_count, 1) * 2 - 1

        self.center_assignation = center_assignation
        self.weight_update = weight_update

        self.beta = - 0.001
        self.bias = 0

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
            rnd_idx = np.random.permutation(len(X))[:self.centers_count]
            self.centers = np.array([X[i, :] for i in rnd_idx])
        elif self.center_assignation == "kmeans":
            self.centers = kmeans(X, self.centers_count, iter=15)[0]

    def _fit_centers_positions(self, X):
        initial_coef = 0.01
        final_coef = 0.001

        epochs = 10

        for epoch in range(epochs):
            change_rate = (initial_coef + epoch / epochs * (final_coef - initial_coef))
            for x in X:
                closest_centre = self.centers[0]
                smallest_dist = euclidean(closest_centre, x)
                for c in self.centers:
                    tmp_dist = euclidean(c, x)
                    if tmp_dist < smallest_dist:
                        smallest_dist = tmp_dist
                        closest_centre = c
                change_vector = (x - closest_centre) * change_rate
                closest_centre += change_vector

    def _update_weights(self, A, Y):
        if self.weight_update == "pinv":
            self.weights = np.linalg.pinv(A).dot(Y)
        elif self.weight_update == "backprop":
            self.weights, self.bias = back_propagation(A, self.weights, Y)
        elif self.weight_update == "pinv + backprop":
            self.weights = np.dot(np.linalg.pinv(A), Y)
            self.weights, self.bias = back_propagation(A, self.weights, Y)

    def train(self, X, Y):
        self._assign_centers(X)
        self._fit_centers_positions(X)
        A = self._calculate_activations(X)
        self._update_weights(A, Y)

    def test(self, X):
        A = self._calculate_activations(X)
        # print(np.dot(A, self.weights) + self.bias)
        return np.dot(A, self.weights) + self.bias
