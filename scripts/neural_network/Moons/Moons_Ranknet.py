#!/usr/bin/env python3

import torch
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    X_train, X_test, y_train, y_test = make_data(n = 1000, create_plot=False)
    # X_train, X_test, y_train, y_test = make_data(n = 1000, create_plot=False)
    input_size = X_train.shape[1]  # This is 2, x1 and x2 plotted on horizontal and vertical
    net = NeuralNetwork_3layer(input_size, 3, 1)
    out = net.forward(X_train)



    print('success')
    print(out)
    return 0

class  NeuralNetwork_3layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.n_input  = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        self.wi = torch.randn(input_size, hidden_size)
        self.wh = torch.randn(hidden_size, output_size)
        self.wo = torch.randn(hidden_size, output_size)

        self.bi = torch.randn(input_size)
        self.bh = torch.randn(hidden_size)
        self.bo = torch.randn(output_size)

    def forward(self, x):
        # First Layer
        x = x.mm(self.wi) # .add(self.bi)
        print(x)
        return x
        print(3)
        return 0
        x = torch.sigmoid(x)
        # Second Layer
        x = torch.mm(self.wh, x)
        x = torch.sigmoid(x)
        # Output Layer
        x = torch.mm(self.wo, x)
        x = torch.sigmoid(x)

        return x


def make_data(create_plot = False, n = 1000):
    #-- Generate Two Moons Data -----------------------------------
    X, y = datasets.make_moons(n_samples = n, noise = 0.3, random_state = 0)
    # Consider reshaping the data
    # y = np.reshape(y, (len(y), 1)) # Make y vertical n x 1 matrix.

    #-- Split data into Training and Test Sets --------------------
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
        torch_data[i] = torch.tensor(data[i], dtype=torch.float, requires_grad=False)

    return torch_data


if __name__ == "__main__":
main()
