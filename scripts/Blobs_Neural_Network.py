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
    
    model.train(X_train, y_train, η=1e-2, iterations=10000)
    plt.plot(model.losses)
    plt.title("Losses at each training iteration")
    plt.show()

    print("The testing misclassification rate is:\n")
    print(model.misclassification_rate(X_test, y_test))


if __name__ == "__main__":
    main()
