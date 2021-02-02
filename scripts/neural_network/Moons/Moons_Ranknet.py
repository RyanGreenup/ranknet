#!/usr/bin/env python3
# * Imports
import torch
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time
import sys

# * Global Variables
DEBUG = False         # Get more verbose printing when trying to debug
dtype = torch.float
if torch.cuda.is_available():  
  print("Detected Cuda Cores, setting Device to Cuda")
  dev = "cuda:0" 
else:  
  print("No cuda cores detected, using CPU")
  dev = "cpu"  

# * Main


def main():
    # TODO The plotting hangs, fix that
    X_train, X_test, y_train, y_test = make_data(n=100000, create_plot=False, noise=0.3)
    # X_train, X_test, y_train, y_test = make_data(n = 1000, create_plot=True)
    input_size = X_train.shape[1]  # This is 2, x1 and x2 plotted on
                                   # horizontal and vertical

    net = NeuralNetwork_2layer(input_size, 3, 1)
    net = net.to(torch.device(dev))

    # TODO why is each batch so slow?
    net.train(X_train, y_train, eta=1e-3, iterations=1e4, batch_size = 10000)

    # print('---\nMisclassification\n')
    # print("Training.........",
    #       round(net.misclassification(X_train, y_train)*100, 2), "%")
    # print("Testing..........",
    #       round(net.misclassification(X_test, y_test)*100, 2), "%")
    return 0


class NeuralNetwork_2layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # inherit the old stuff, I think TODO Clarify this
        # Weights
        self.wi = torch.nn.Parameter(torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True, device = dev))
        self.wo = torch.nn.Parameter(torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True, device = dev))

        # Biases
        self.bi = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype, requires_grad=True, device = dev))
        self.bo = torch.nn.Parameter(torch.randn(output_size, dtype=dtype, requires_grad=True, device = dev))

        # Loss Function and list
        self.loss_fn = torch.nn.MSELoss()
        self.losses = []  # Losses at each iteration

    def forward(self, xi, xj):
        si = self.forward_single(xi)
        sj = self.forward_single(xj)

        sigma = 1  # TODO this should be a tensor variable with a gradient
        return torch.sigmoid(sigma*(si-sj)).to(dev)

    def forward_single(self, x):
        # First Layer
        x = torch.mm(x, self.wi).add(self.bi)
        x = torch.sigmoid(x)
        # Output Layer
        x = x.mm(self.wo).add(self.bo)
        x = torch.sigmoid(x)
        x = x.to(dev)

        return x

    @staticmethod
    def rank_encode(xi, xj):
        if xi > xj:
            return 1
        elif xi == xj:
            return 0
        else:
            return -1

    def train(self, x, y, eta, iterations, batch_size):
        opt = torch.optim.RMSprop(self.parameters(), lr=eta)
        for t in range(int(iterations)):
            start = time.time()

            samples = np.array([random.sample(range(x.shape[0]), 2) for i in range(batch_size)])
            xi = x[samples[:, 0], :]
            xj = x[samples[:, 1], :]
            yi = y[samples[:, 0], :]
            yj = y[samples[:, 1], :]

            # TODO This is the issue .................................
            #      Why is this so slow? This scales in linear time
            #      with respect to the batch size
            y_batch_np = ((yi>yj)*2 - 1)*(yi != yj)
            y_batch = torch.tensor(y_batch_np, dtype=dtype, requires_grad=False)

            # Make y vertical n x 1 matrix to match network output
            y_batch = torch.reshape(y_batch, (len(y_batch), 1), )
            # .........................................................
            if DEBUG:
                print(yi)
                print(xi)
                print(yj)
                print(xj)
                print('---')
                print(y_batch)

            stop = time.time()
            print(stop - start)
            sys.exit(0)
            # Make the Prediction; Forward Pass
            y_pred = self.forward(xi, xj)


            # Measure the loss
            loss = self.loss_fn(y_pred, y_batch)  # input, target is correct order
            self.losses.append(loss.item())
            if t%100:
                print(t*100/iterations, "%")

            # Backwards Pass
            # First Zero the Gradients, otherwise they can't be overwritten
            opt.zero_grad()

            # Calculate the Gradients
            loss.backward()

            # Adjust the Weights
            opt.step()

        np.savetxt("lossesOut.txt", np.array(self.losses), delimiter=",")
        plt.plot(self.losses)
        plt.show()

    def misclassification(self, x, y):
        y_pred = self.forward(x) > 0.5
        return np.average(y_pred != y)


def make_data(create_plot=False, n=1000, noise=0.3):
    # -- Generate Two Moons Data -----------------------------------
    # In this case the bottom right (1) is relevant data,
    # top left (0) is not relevant data, bottor right is rel (1)
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=0)
    # Consider reshaping the data
    y = np.reshape(y, (len(y), 1))  # Make y vertical n x 1 matrix.

    # -- Split data into Training and Test Sets --------------------
    # X_train, X_test, y_train, y_test
    data = train_test_split(X, y, test_size=0.4)

    if(create_plot):
        # Make an empty figure
        p = plt.figure()

        # Create the Scatter Plot
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.legend()
        p.show()

    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False, device = dev)

    return torch_data


# if __name__ == "__main__":
main()




# Notes
    # Need to use
        # def __init__(self):
        #     super().__init__()
        #     self.losses = []

    # All parameters MUST be wrapped in torch.nn.parameters()
