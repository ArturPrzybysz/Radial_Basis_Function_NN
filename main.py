from rbfn import rbfn
from matplotlib import pyplot as plt
from learning_data_import import *

# X, Y = custom_function1(300)
X, Y = school_data_set()

# weight_update options: ["pinv", "backprop", "pinv + backprop"]
# center_assignation options: [kmeans", "random"]

net = rbfn(input_size=1,
           centers_count=10,
           center_assignation="random",
           # center_assignation="kmeans",
           weight_update="pinv")
           # weight_update="backprop")
           # weight_update="pinv + backprop")

net.train(X, Y)

Z = net.test(X)
plt.plot(X, Y, 'k-')
plt.plot(X, Z, 'r-', linewidth=1.5)
plt.plot(net.centers, np.zeros(net.centers_count), 'cs')
plt.show()
