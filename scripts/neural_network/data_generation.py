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