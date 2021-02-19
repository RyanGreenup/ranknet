#!/usr/bin/env python

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import random
from ranknet.test_cuda import test_cuda
from ranknet.make_data import make_data
from ranknet.neural_network import three_layer_ranknet_network
from ranknet.quicksort import quicksort

# Set Seeds
torch.manual_seed(1)
np.random.seed(1)

# Set Torch Parameters
dtype = torch.float
dev = test_cuda()

# Set personal flags
DEBUG = True


# Main Function

def main():
    # Make the Data
    X_train, X_test, y_train, y_test = make_data(
        n=int(300/0.4), create_plot=True, dtype=dtype, dev=dev)

    # Create a model object
    model = three_layer_ranknet_network(
        input_size=X_train.shape[1], hidden_size=2, output_size=1, dtype=dtype, dev=dev)

    # Train the Model
    model.train(X_train, y_train, η=1e-1, iterations=1e2, batch_size=30)

    # Visualise the Training Error
    plot_losses(model)

    # Misclassification won't work for ranked data
    # Instead Visualise the ranking
    plot_ranked_data(X_test, y_test, model)


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


if __name__ == "__main__":
    main()
