from rbfn import rbfn
import numpy as np
from matplotlib import pyplot as plt

n = 300

X = np.mgrid[0:4:complex(0, n)].reshape(n, 1)
Y = np.array([np.sin(x) ** 2 * np.exp(x / 100) for x in X])

# weight_update options: ["pinv", "backprop", "pinv + backprop"]

net = rbfn(input_size=1,
           output_size=1,
           centers_count=5,
           weight_update="backprop")

net.train(X, Y)

Z = net.test(X)

plt.plot(X, Y, 'k-')
plt.plot(X, Z, 'r-', linewidth=2)
plt.plot(net.centers, np.zeros(net.centers_count), 'cs')

plt.show()
