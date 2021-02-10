#!/usr/bin/env python
# TODO Train the network using only the 1-1 comparisons
# TODO Implement the factoring trick to improve performance
# TODO Implement a way to rank each value, then plot that ranking to the plot.



# Import Packages
from sklearn import datasets
import numpy as np
from progress.bar import Bar
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn
from itertools import tee
import random

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
    X_train, X_test, y_train, y_test = make_data(n = 30, create_plot=False)
    model = three_layer_nn(input_size=X_train.shape[1], hidden_size=2, output_size=1)
    out1 = model.forward(X_train[1,:], X_train[2,:])
    model.train(X_train, y_train)
    # print("Training Error: ",  100*model.misclassification_rate(X_train, y_train), "%")
    # print("Testing Error: ", 100*model.misclassification_rate(X_test, y_test), "%")
    plt.plot(model.losses)

    if DEBUG:
        report_val(X_train, y_train, model)
    
    plt.show()
    sys.exit(0)


class three_layer_nn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(three_layer_nn, self).__init__()
        self.wi = torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True, device = dev)
        self.wo = torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True, device = dev)
                                                                                      
        self.bi = torch.randn(hidden_size, dtype=dtype, requires_grad=True, device = dev)
        self.bo = torch.randn(output_size, dtype=dtype, requires_grad=True, device = dev)
                                                                                      
        self.σ = torch.randn(1, dtype=dtype, requires_grad=True, device = dev)

        self.losses = []

    def forward(self, xi, xj):
        si = self.forward_single(xi)
        sj = self.forward_single(xj)
        out = 1/(1+torch.exp(-self.σ*(si-sj))) #0x3c3
        return out
    def forward_single(self, x):
        x = torch.matmul(x, self.wi).add(self.bi)
        x = torch.matmul(x, self.wo).add(self.bo)
        return x

    def loss_fn(self, xi, xj, y):
        y_pred = self.forward(xi, xj)
        return torch.mean(torch.pow((y-y_pred), 2))

    def misclassification_rate(self, x, y):
        y_pred = (self.forward(x) > 0.5)
        return np.average(y != y_pred)

    def train(self, x, target, η=1e-4, iterations=1e3):
        bar = Bar('Processing', max=iterations)
        for t in range(int(iterations)):
            sublosses = []
            for pair in pairwise(range(len(x)-1)):
                sublosses = []
                xi =      x[pair[0],]
                yi = target[pair[0]]
                xj =      x[pair[1],]
                yj = target[pair[1]]

                # rencode from {0, 1} to {-1, 0, 1}
                y = ((yi > yj)*2 - 1)*(yi != yj)

                # Calculate y, forward pass
                y_pred = self.forward(xi, xj)

                # Measure the loss
                loss = self.loss_fn(xi, xj, y)
                # print(loss.item())
                sublosses.append(loss.item())

                # Calculate the Gradients with Autograd
                loss.backward()

                with torch.no_grad():
                    # Update the Weights with Gradient Descent 
                    self.wi -= η * self.wi.grad; self.wi.grad = None
                    self.bi -= η * self.bi.grad; self.bi.grad = None
                    self.wo -= η * self.wo.grad; self.wo.grad = None
                    self.bo -= η * self.bo.grad; self.bo.grad = None
                    self.σ  -= η * self.σ.grad; self.σ.grad   = None

                    # ; Zero out the gradients, they've been used

            self.losses.append(np.average(sublosses))
            bar.next()
        bar.finish()


def pairwise(iterable): # NOTE https://docs.python.org/3/library/itertools.html
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)





def make_data(create_plot=False, n=100):
    # -- Generate Two Moons Data -----------------------------------
    # X, y = datasets.make_moons(n_samples=n, noise=0.1, random_state=0) # Top left is 0, # Bottom Right is 1
    temp_X, y = datasets.make_blobs(100, 2, 2, random_state=7) # Yellow is relevant
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