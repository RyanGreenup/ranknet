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
from ranknet.quicksort_arr import quicksort


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
    X_train, X_test, y_train, y_test = make_data(n=100, create_plot=True)
    model = three_layer_nn(
        input_size=X_train.shape[1], hidden_size=2, output_size=1, dtype=dtype, dev=dev)
    out1 = model.forward(X_train[1, :], X_train[2, :])
    model.train(X_train, y_train, iterations=5e2)

    plot_losses(model)
    plot_ranked_data(X_test, y_test, model)


    sys.exit(0)

def plot_losses(model):
    plt.plot(model.losses)
    plt.title("Cost / Loss Function for Iteration of Training")
    plt.show()

def plot_ranked_data(X, y, model):
    # Create a list of values
    n = X.shape[0]
    order = [i for i in range(n)]
    # Arrange that list of values based on the model
    quicksort(values=order, left=0, right=(n-1), data=X, model=model)
    print(order)

    ordered_data = X[order, :]
    y_ordered = y[order]

    p = plt.figure()
    for i in range(len(ordered_data)):
        plt.text(ordered_data[i, 0], ordered_data[i, 1], i)
    plt.scatter(ordered_data[:, 0], ordered_data[:, 1], c=y_ordered)
    plt.title("Testing Data, with ranks")
    plt.show()


def make_data(create_plot=False, n=1000):
    # -- Generate Two Moons Data -----------------------------------
    # Top left is 0, # Bottom Right is 1
    # temp_X, y = datasets.make_moons(n_samples=n, noise=0.1, random_state=0)
    temp_X, y = datasets.make_blobs(n, 2, 2, random_state=7) # Yellow is relevant
    # Rotate the plot 90 deg
    X = np.ndarray(temp_X.shape)
    X[:, 0] = temp_X[:, 1]
    X[:, 1] = temp_X[:, 0]
    # Consider reshaping the data
    y = np.reshape(y, (len(y), 1))  # Make y vertical n x 1 matrix.

    # -- Split data into Training and Test Sets --------------------
    # X_train, X_test, y_train, y_test
    data = train_test_split(X, y, test_size=0.4)

    if(create_plot):
        # Create the Scatter Plot
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Sample Data")
        plt.show()

    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False)

    return torch_data

main()
