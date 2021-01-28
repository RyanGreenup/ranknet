#!/usr/bin/env python3

import torch
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dtype = torch.float

def main():
    X_train, X_test, y_train, y_test = make_data(n = 1000, create_plot=False)
    # X_train, X_test, y_train, y_test = make_data(n = 1000, create_plot=False)
    input_size = X_train.shape[1]  # This is 2, x1 and x2 plotted on horizontal and vertical
    net = NeuralNetwork_2layer(input_size, 3, 1)

    net.train(X_train, y_train, eta=1e-2)

    print('---\nMisclassification\n')
    print("Training.........", round(net.misclassification(X_train, y_train)*100, 2), "%")
    print("Testing..........", round(net.misclassification(X_test, y_test)*100, 2), "%")
    return 0

class  NeuralNetwork_2layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # inherit the old stuff, I think TODO Clarify this
        # Weights
        self.wi = torch.nn.Parameter(torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True))
        self.wo = torch.nn.Parameter(torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True))

        # Biases
        self.bi = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype, requires_grad=True))
        self.bo = torch.nn.Parameter(torch.randn(output_size, dtype=dtype, requires_grad=True))

        # Loss Function and list
        self.loss_fn  = torch.nn.MSELoss()
        self.losses = [] # Losses at each iteration

    def forward(self, x):
        # First Layer
        x = torch.mm(x, self.wi).add(self.bi)
        x = torch.sigmoid(x)
        # Output Layer
        x = x.mm(self.wo).add(self.bo)
        x = torch.sigmoid(x)

        return x

    def train(self, x, y, eta):
        opt = torch.optim.RMSprop(self.parameters(), lr = eta)
        for t in range(int(1e3)):

            # Make the Prediction; Forward Pass
            y_pred = self.forward(x)

            # Measure the loss
            loss = self.loss_fn(y_pred, y) # input, target is correct order
            if t%100:
                print(loss.item())

            # Backwards Pass
            # First Zero the Gradients, otherwise they can't be overwritten
            opt.zero_grad()

            # Calculate the Gradients
            loss.backward()

            # Adjust the Weights
            opt.step()

    def misclassification(self, x, y):
        y_pred = self.forward(x) > 0.5
        return np.average(y_pred != y)



def make_data(create_plot=False, n=1000):
    # -- Generate Two Moons Data -----------------------------------
    X, y = datasets.make_moons(n_samples=n, noise=0.3, random_state=0) # Top left is 0, # Bottom Right is 1
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


if __name__ == "__main__":
main()


# Notes
    # Need to use
        # def __init__(self):
        #     super().__init__()
        #     self.losses = []

    # All parameters MUST be wrapped in torch.nn.parameters()
