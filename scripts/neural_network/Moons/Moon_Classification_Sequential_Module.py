#!/usr/bin/env python

# TODO Perhaps this approach actually worked and I was measuring the misclassification rate incorrectly?
# TODO Perhaps the Loss function is why this approach performed so poorly?
# TODO Implement testing and training data
  # TODO Implement a plot of both Testing and Training Data
# TODO Implement a method to intelligently exit the loop

#-- Import Packages -------------------------------------------
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt

#-- Generate Two Moons Data -----------------------------------
X, y = datasets.make_moons(n_samples = 1000, noise = 0.3, random_state = 0)
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
device = 'cpu'
dtype = float
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))

## Define a model
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=3),
    nn.Sigmoid(),
    nn.Linear(3, 3),
    nn.Sigmoid(),
    nn.Linear(3, 1),
    # nn.Sigmoid(),
)

## Define a Loss Function
loss_fn = torch.nn.BCEWithLogitsLoss()
## Define an Optimizer
eta = 1e-1
eta = 0.005
optimizer = torch.optim.RMSprop(model.parameters(), lr = eta)
## Train the Model
losses = []
for t in range(10000):
    # Forward Pass: Compute predicted y value
    y_pred = model(X.float())

    # Measure the Loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, '\t', loss.item())
    # Log the Loss
    losses.append(loss.item())

    # Backward Pass; Compute the Partial Derivatives
    ## First Zero the Gradients, otherwise the can't be overwritten
    optimizer.zero_grad()
    
    ## Now calculate the gradients
    loss.backward()

    # Adjust the Weights
    optimizer.step()
    

# Measure the Misclassification Rate ------------------------
yhat = model.forward(X).detach().numpy().reshape((1, -1))
yhat = [i>0.5 for i in yhat ]
yhat = np.array(yhat).astype(int)
y = y.detach().numpy().reshape(1, -1).astype(int)

import tools
misclassification_rate_nn = tools.misclassification_rate(yhat, y)
misclassification_rate_tree = tools.misclassification_rate(clf.predict(X), y)

# misclassification_rate_nn = np.average(yhat == y.reshape(1, -1))
# [i == j for i in yhat for j in y.reshape(1, -1)]

print("The misclassification rate tree is:\n", misclassification_rate_tree)
print("The misclassification rate network is:\n", misclassification_rate_nn)


# print("Final loss: ", losses[-1])
plt.plot(losses)
plt.show()