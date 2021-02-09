#!/usr/bin/env python
# TODO Train the network using only the 1-1 comparisons
# TODO Implement the factoring trick to improve performance



# Import Packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn

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

# Main Function
def main():
    X_train, X_test, y_train, y_test = make_data(n = 30, create_plot=True)
    model = three_layer_nn(input_size=X_train.shape[1], hidden_size=2, output_size=1)
    out1 = model.forward(X_train)
    model.train(X_train, y_train)
    print("Training Error: ",  100*model.misclassification_rate(X_train, y_train), "%")
    print("Testing Error: ", 100*model.misclassification_rate(X_test, y_test), "%")
    plt.plot(model.losses)
    plt.show()
    sys.exit(0)


class three_layer_nn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(three_layer_nn, self).__init__()
        self.wi = torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True)
        self.wo = torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True)

        self.bi = torch.randn(hidden_size, dtype=dtype, requires_grad=True)
        self.bo = torch.randn(output_size, dtype=dtype, requires_grad=True)

        self.losses = []

    def forward(self, x):
        x = torch.mm(x, self.wi).add(self.bi)
        x = torch.mm(x, self.wo).add(self.bo)
        return x

    def loss_fn(self, x, y):
        y_pred = self.forward(x)
        return torch.mean(torch.pow((y-y_pred), 2))

    def misclassification_rate(self, x, y):
        y_pred = (self.forward(x) > 0.5)
        return np.average(y != y_pred)

    def train(self, x, target, η=1e-4, iterations=2e4):
        for t in range(int(iterations)):
            if (t % 1000 == 0):
                print(t*100/iterations, "%")
            # Calculate y, forward pass
            y_pred = self.forward(x)

            # Measure the loss
            loss = self.loss_fn(x, target)
            # print(loss.item())
            self.losses.append(loss.item())

            # Calculate the Gradients with Autograd
            loss.backward()

            with torch.no_grad():
                # Update the Weights with Gradient Descent 
                self.wi -= η * self.wi.grad; self.wi.grad = None
                self.bi -= η * self.bi.grad; self.bi.grad = None
                self.wo -= η * self.wo.grad; self.wo.grad = None
                self.bo -= η * self.bo.grad; self.bo.grad = None

                # ; Zero out the gradients, they've been used







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
        # Create the Scatter Plot
        plt.scatter(X[:,0], X[:,1], c = y)
        plt.show()

    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False)

    return torch_data

main()