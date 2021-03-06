#!/usr/bin/env python3

import sys
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import sklearn
import numpy as np
import torch
DEBUG = False         # Get more verbose printing when trying to debug

torch.manual_seed(329832)  # set the seed.
np.random.seed(39832)

# * Global Variables
DEBUG = False         # Get more verbose printing when trying to debug
dtype = torch.float
# if torch.cuda.is_available():
#   print("Detected Cuda Cores, setting Device to Cuda")
#   dev = "cuda:0"
# else:
#   print("No cuda cores detected, using CPU")
#   dev = "cpu"
dev = "cpu"

# * Main


def main():
    X_train, X_test, y_train, y_test = make_data(
        n=100, create_plot=True, noise=0.1)
    input_size = X_train.shape[1]  # This is 2, x1 and x2 plotted on
    # horizontal and vertical

    net = NeuralNetwork_2layer(input_size, 3, 1)
    net = net.to(torch.device(dev))

    net.train(X_train, y_train, eta=0.5*1e-2, iterations=1*1e3)

    return 0


class NeuralNetwork_2layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # inherit the old stuff, I think TODO Clarify this
        # Weights
        self.wi = torch.nn.Parameter(torch.randn(
            input_size, hidden_size, dtype=dtype, requires_grad=True))
        self.wo = torch.nn.Parameter(torch.randn(
            hidden_size, output_size, dtype=dtype, requires_grad=True))

        # Biases
        self.bi = torch.nn.Parameter(torch.randn(
            hidden_size, dtype=dtype, requires_grad=True))
        self.bo = torch.nn.Parameter(torch.randn(
            output_size, dtype=dtype, requires_grad=True))

        # Sigma Value
        self.sigma = torch.nn.Parameter(torch.randn(
            1, dtype=dtype, requires_grad=True))

        # Loss Function and list
        self.losses = []  # Losses at each iteration
        self.mcr_list = []  # Misclassification Rate

    def forward(self, xi, xj):
        si = self.forward_single(xi)
        sj = self.forward_single(xj)

        return torch.sigmoid(self.sigma*(si-sj))

    def forward_single(self, x):
        # First Layer
        x = torch.mm(x, self.wi).add(self.bi)
        x = torch.sigmoid(x)
        # Output Layer
        x = x.mm(self.wo).add(self.bo)
        x = torch.sigmoid(x)

        return x

    @staticmethod
    def BCE_wide(Pij_pred, Pij_actual):
        return torch.mean(-Pij_actual*torch.log(Pij_pred) - (1-Pij_actual)*torch.log(1-Pij_pred))

    def make_samples(self, x, y, batch_size, Boolean_Range):
        """
        Boolean_Range is Whether to use {0, 1} rather than {-1, 0, 1} as range
        """
        samples = np.array([random.sample(range(x.shape[0]), 2)
                            for i in range(batch_size)])
        xi = x[samples[:, 0], :]
        xj = x[samples[:, 1], :]
        yi = y[samples[:, 0], :]
        yj = y[samples[:, 1], :]

        if (Boolean_Range):
            self.loss_fn = torch.nn.BCELoss()  # NOTE targets should be y \in [0, 1] for this
            y_batch = torch.tensor([int(yi[k] > yj[k]) for k in range(
                batch_size)], dtype=dtype, requires_grad=False)
            # Make y vertical n x 1 matrix to match network output
            y_batch = torch.reshape(y_batch, (len(y_batch), 1))
        else:
            self.loss_fn = self.BCE_wide  # NOTE targets should be y \in [0, 1] for this
            # Measure whether or not i is greater than j
            y_batch_boolean = yi > yj
            # rencode from {0, 1} to {-1, 0, 1}
            y_batch_np = ((yi > yj)*2 - 1)*(yi != yj)
            # Make it a tensor
            y_batch = torch.tensor(
                y_batch_np, dtype=dtype, requires_grad=False)
            # Make y vertical n x 1 matrix to match network output
            y_batch = torch.reshape(y_batch, (len(y_batch), 1))
        if DEBUG:
            print(yi)
            print(xi)
            print(yj)
            print(xj)
            print('---')
            print(y_batch)

        return xi, xj, yi, yj, y_batch

    def train(self, x, y, eta, iterations):
        opt = torch.optim.RMSprop(self.parameters(), lr=eta)
        for t in range(int(iterations)):

            xi, xj, yi, yj, y_batch = self.make_samples(
                x, y, batch_size=int(10*1e3), Boolean_Range=1)

            # Make the Prediction; Forward Pass
            y_pred = self.forward(xi, xj)

            # Measure the loss
            # input, target is correct order
            loss = self.loss_fn(y_pred, y_batch)
            self.losses.append(loss.item())
            if t % 100:
                print(t*100/iterations, "%")

            # Also measure the Misclassification Rate (because it's more intuitive)
            m_rate = np.average((y_pred > 0.5) != y_batch)
            self.mcr_list.append(m_rate)

            # Backwards Pass
            # First Zero the Gradients, otherwise they can't be overwritten
            opt.zero_grad()

            # Calculate the Gradients
            loss.backward()

            # Adjust the Weights
            opt.step()

        # plt.plot(self.losses)
        plt.plot(self.mcr_list)
        plt.show()


def make_data(create_plot=False, n=1000, noise=0.3):
    # -- Generate Two Moons Data -----------------------------------
    # In this case the bottom right (1) is relevant data,
    # top left (0) is not relevant data
    # Top left is 0, # Bottom Right is 1
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=0)
    # Consider reshaping the data
    y = np.reshape(y, (len(y), 1))  # Make y vertical n x 1 matrix.

    # -- Split data into Training and Test Sets --------------------
    # X_train, X_test, y_train, y_test
    data = train_test_split(X, y, test_size=0.4)

    if(create_plot):
        # Create the Scatter Plot
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()

    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False)

    return torch_data


if __name__ == "__main__":
    main()


# Notes
    # Need to use
    # def __init__(self):
    #     super().__init__()
    #     self.losses = []

    # All parameters MUST be wrapped in torch.nn.parameters()
