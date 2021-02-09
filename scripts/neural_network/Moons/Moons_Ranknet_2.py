#!/usr/bin/env python
# Import Packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

dtype = torch.float

# Main Function
def main():
    X_train, X_test, y_train, y_test = make_data(n = 30, create_plot=True)
    



def make_data(create_plot=False, n=100):
    # -- Generate Two Moons Data -----------------------------------
    X, y = datasets.make_moons(n_samples=n, noise=0.1, random_state=0) # Top left is 0, # Bottom Right is 1
    X, y = datasets.make_blobs(100, 2, 2, random_state=7)
    # Consider reshaping the data
    y = np.reshape(y, (len(y), 1)) # Make y vertical n x 1 matrix.

    # -- Split data into Training and Test Sets --------------------
    # X_train, X_test, y_train, y_test
    data = train_test_split(X, y, test_size = 0.4)

    if(create_plot):
        # Make an empty figure
        p = plt.figure()
        # Create the Scatter Plot
        plt.scatter(X[:,0], X[:,1], c = y)
        p.show()

    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False)

    return torch_data

main()