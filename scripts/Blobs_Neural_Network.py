#!/usr/bin/env python

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import random
from ranknet.test_cuda import test_cuda
from ranknet.make_data import make_data
from ranknet.neural_network import three_layer_classification_network

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
        n=100, create_plot=True, dtype=dtype, dev=dev)

    # Create a model object
    model = three_layer_classification_network(
        input_size=X_train.shape[1], hidden_size=2, output_size=1, dtype=dtype, dev=dev)


    # Send some data through the model
    print("\nThe Network input is:\n---\n")
    print(X_train[7,:], "\n")
    print("The Network Output is:\n---\n")
    print(model.forward(X_train[7,:]).item(), "\n")


if __name__ == "__main__":
    main()
