
#-- Import Packages -------------------------------------------
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt

#-- Generate Two Moons Data -----------------------------------
X, y = datasets.make_moons(n_samples = 400, noise = 0.4, random_state = 3141, shuffle=True)
y = np.reshape(y, (len(y), 1)) # Make y vertical n x 1 matrix.

# Plot the Generated Data -----------------------------------
    # Make an empty figure
# plt.ion()
p = plt.figure()
    # Create the Scatter Plot
plt.scatter(X[:,0], X[:, 1], c = y)
    # Labels
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Plot of Two Moons Data")
    # Show the Plot
plt.show()

#-- Split data into Training and Test Sets --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

#-Model the Data Using Trees---------------------------------
#  Define a Decision Tree Classifier -------------------------
d = 4
clf = sklearn.tree.DecisionTreeClassifier(max_depth = d)

#. Train the Model ..........................................
clf.fit(X_train, y_train)

#. Test the Model
score = clf.score(X_test, y_test) # mean accuracy (TP+TN)/(P+N)
misclassification_rate_tree = np.average(clf.predict(X) == y)

print("The performance is:\n" + str(score*100) + "%")
print("The misclassification rate is:\n", misclassification_rate_tree)

# Fit the Neural Network ------------------------------------
## Import the Packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
## Create Tensors
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
## Define a Model as a class
input_size = 2
hidden_size = 3 # This is arbitrary
output_size = 1 # should be 1, just like y

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size):

        # Initialize Random Weights
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, hidden_size, requires_grad=True)
        self.W3 = torch.randn(hidden_size, output_size, requires_grad=True)

        # Add Bias Values
        self.b1 = torch.randn(hidden_size, requires_grad = True)
        self.b2 = torch.randn(hidden_size, requires_grad = True)
        self.b3 = torch.randn(output_size, requires_grad = True)

    # Define the Forward Pass
    def forward(self, inputs):
        z1 = inputs.mm(self.W1).add(self.b1)
        a1 = 1 / (1 + torch.exp(-z1))
        z2 = a1.mm(self.W2).add(self.b2)
        a2 = 1/(1+torch.exp(-z2))
        z3 = a2.mm(self.W3).add(self.b3)
        output = 1/(1+torch.exp(-z3))
        return output


input_size = 2 
hidden_size = 3 # randomly chosen
output_size = 1 # we want it to return a number that can be used to calculate the difference from the actual numberclass NeuralNetwork():

epochs = 10000
learning_rate = 0.005
model = NeuralNetwork(input_size, hidden_size, output_size)
inputs = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.float)#store all the loss values
losses = []


for epoch in range(epochs):
    # Forward Pass (Calculate y_pred)
    output = model.forward(inputs)

    # Binary Cross Entropy Formula
    loss = -((labels * torch.log(output)) + (1 - labels) * torch.log(1 - output)).sum()

    # Log the log so it can be plotted after the fact
    losses.append(loss.item())

    # Calculate the gradients of the weights wrt to loss
    loss.backward()
    print(loss.item())

    # Adjust the weights based on the previous calculated gradients
    model.W1.data -= learning_rate * model.W1.grad
    model.W2.data -= learning_rate * model.W2.grad
    model.W3.data -= learning_rate * model.W3.grad
    model.b1.data -= learning_rate * model.b1.grad
    model.b2.data -= learning_rate * model.b2.grad
    model.b3.data -= learning_rate * model.b3.grad

    # Zero the Gradients so they aren't readjusted
    model.W1.grad.zero_()
    model.W2.grad.zero_()
    model.W3.grad.zero_()
    model.b1.grad.zero_()
    model.b2.grad.zero_()
    model.b3.grad.zero_()

# print("Final loss: ", losses[-1])
plt.plot(losses)
plt.show()

# Measure the Misclassification Rate ------------------------
yhat = model.forward(X).detach().numpy().reshape((1, -1))
yhat = [i>0.5 for i in yhat ]
yhat = np.array(yhat).astype(int)
y = y.detach().numpy().reshape(1, -1).astype(int)

import tools
misclassification_rate_nn = tools.misclassification_rate(yhat, y)
misclassification_rate_tree = tools.misclassification_rate(clf.predict(X), y)


print("The misclassification rate for the tree based model is:\n", misclassification_rate_tree)
print("The misclassification rate for the neural network model is:\n", misclassification_rate_nn)
