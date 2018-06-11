from rbfn import rbfn
from learning_data_import import *

X, Y = classification_learning_set()
X_test, Y_test = classification_test_set()

# weight_update options: ["pinv", "backprop", "pinv + backprop"]
# center_assignation options: [kmeans", "random"]

net = rbfn(input_size=4,
           centers_count=2,
           center_assignation="random",
           # center_assignation="kmeans",
           # weight_update="pinv")
           weight_update="backprop")
# weight_update="pinv + backprop")

net.train(X, Y)

Z = net.test(X_test)

correct_guess = 0
guess_count = 0

for actual, expected in zip(Z, Y_test):
    correct_guess += np.round(actual) == expected
    guess_count += 1

print(100 * correct_guess / guess_count, "%")

for i in range(len(Z)):
    if Z[i] > 3:
        Z[i] = 3
    elif Z[i] < 0:
        Z[i] = 0

confusion_matrix = np.zeros((3, 3))
for i in range(len(Z)):
    confusion_matrix[int(np.round(Z[i])) - 1][int(np.round(Y_test[i])) - 1] += 1

print(confusion_matrix)
