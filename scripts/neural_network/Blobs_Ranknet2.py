#!/usr/bin/env python
# TODO Implement the factoring trick to improve performance
# TODO Implement a way to rank each value, then plot that ranking to the plot, with numbers



# Import Packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn
import random
from ranknet.three_layer_nn_class import three_layer_nn


torch.manual_seed(1)  # set the seed.
np.random.seed(1)

dtype = torch.float


# if torch.cuda.is_available():
#   print("Detected Cuda Cores, setting Device to Cuda")
#   dev = "cuda:0"
# else:
#   print("No cuda cores detected, using CPU")
#   dev = "cpu"
dev = "cpu"
DEBUG = True

# Main Function
def main():
    X_train, X_test, y_train, y_test = make_data(n = 200, create_plot=True)
    model = three_layer_nn(input_size=X_train.shape[1], hidden_size=2, output_size=1, dtype=dtype, dev=dev)
    out1 = model.forward(X_train[1,:], X_train[2,:])
    model.train(X_train, y_train)
    # print("Training Error: ",  100*model.misclassification_rate(X_train, y_train), "%")
    # print("Testing Error: ", 100*model.misclassification_rate(X_test, y_test), "%")
    plt.plot(model.losses)
    plt.title("Cost / Loss Function for Iteration of Training")
    plt.show()
    plt.title("blah")

    if DEBUG:
        report_val(X_train, y_train, model)
    
    print("The Training Misclassification Rate is: ", model.misclassification_rate(X_train, y_train, model.threshold))
    print("The Testing Misclassification Rate is: ", model.misclassification_rate(X_test, y_test, model.threshold))
    
    sys.exit(0)






def make_data(create_plot=False, n=1000):
    # -- Generate Two Moons Data -----------------------------------
    # X, y = datasets.make_moons(n_samples=n, noise=0.1, random_state=0) # Top left is 0, # Bottom Right is 1
    temp_X, y = datasets.make_blobs(n, 2, 2, random_state=7) # Yellow is relevant
    # Rotate the plot 90 deg
    X = np.ndarray(temp_X.shape)
    X[:,0] = temp_X[:,1]
    X[:,1] = temp_X[:,0]
    # Consider reshaping the data
    y = np.reshape(y, (len(y), 1)) # Make y vertical n x 1 matrix.

    # -- Split data into Training and Test Sets --------------------
    # X_train, X_test, y_train, y_test
    data = train_test_split(X, y, test_size = 0.4)

    if(create_plot):
        # Create the Scatter Plot
        plt.scatter(X[:,0], X[:,1], c = y)
        plt.title("Sample Data")
        plt.show()

    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False)

    return torch_data


def report_val(X, y, model):
    vals = random.sample(range(len(X)-1), 2)
    xi = X[vals[0],:]
    yi = y[vals[0]]
    xj = X[vals[1],:]
    yj = y[vals[1]]

    y_pred = model.forward(xi, xj)
    y = ((yi > yj)*2 - 1)*(yi != yj)

    print("so for the two points:")
    print(xi)
    print(xj)
    print("The Actual state of the first one being ranked higher is")
    print(y)
    print("The model returns the value")
    print(y_pred)


main()