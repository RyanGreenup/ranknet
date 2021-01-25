#!/usr/bin/env python
import os, sys
# TODO calculate loss_fn with respect to batches of length 100-1000
# TODO Implement an optimizer like RMSProp
  # This could implement it's own batching, so adjust the batch size to be as large as possible
# os.chdir(os.path.dirname(sys.argv[0]))
# os.chdir("/home/ryan/Studies/2020ResearchTraining/ranknet/scripts/neural_network/")

# * Import Packages
#   ___                            _
#  |_ _|_ __ ___  _ __   ___  _ __| |_
#   | || '_ ` _ \| '_ \ / _ \| '__| __|
#   | || | | | | | |_) | (_) | |  | |_
#  |___|_| |_| |_| .__/ \___/|_|   \__|
#                |_|
# ** Typical stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# ** Torch Stuff
import torch
import random

# * Main
#  __  __       _
# |  \/  | __ _(_)_ __
# | |\/| |/ _` | | '_ \
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|
#

def main():
    # Load Data ................................................
    X, X_test, y, y_test = load_data()
    print(X.shape[0])

    # Initialise a Model Object ..................................
    net = NeuralNetwork()
    print('The Neural Network is described as:\n')
    print(net)

    # Choose a Loss Function  ....................................
    def loss_fn_MSE(output, target):
        loss = torch.mean((output-target)**2)
        # loss = torch.mean(output**(1-target)*(1-output)**target)
        return loss

    def loss_fn(S_ij, P_ij): # S_ij = pbar
        loss = torch.mean(-S_ij * torch.log(P_ij) - (1-S_ij)*torch.log(1-P_ij))
        return loss

    # loss_fn = torch.nn.MSELoss()

    eta = 1e-6

    # Choose an Optimizer
    # optimizer = torch.optim.RMSprop(net.parameters(), lr = eta)
    # DONE Consider here, not actually using any optimizer and adjusting manually
    optimizer = torch.optim.SGD(net.parameters(), lr = eta)

    # Train the Model .............................................
    net.train_model(eta, loss_fn, X, y)

    # Measure the misclassification rate....................
    # TODO Adapt this for ranknet
    # m = misclassification_rate(X, X_test, y, y_test, net.forward)
    # m.report()

    # plot the losses.......................................
    plt.plot(net.losses)
    plt.show()


#  _                    _   ____        _
# | |    ___   __ _  __| | |  _ \  __ _| |_ __ _
# | |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
# | |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
# |_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|

def load_data(datafile='./DataSets/winequality-red.csv'):
    df=pd.read_csv(datafile, sep=';')
    df=np.array(df.values)

    # Extract the Features
    y = df[:,-1]
    X = df[:,range(df.shape[1]-1)]

    # Make the data categorical
    y = y>5

    # Transfom the Data into Tensors
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))

    #-- Split data into Training and Test Sets --------------------
    X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1)
    return X, X_test, y, y_test


#  _   _                      _   _   _      _                      _
# | \ | | ___ _   _ _ __ __ _| | | \ | | ___| |___      _____  _ __| | __
# |  \| |/ _ \ | | | '__/ _` | | |  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ /
# | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   <
# |_| \_|\___|\__,_|_|  \__,_|_| |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\

# Define the Class for Torch
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []

        # Inputs to hidden layer linear transformation
        # Create the input layer and the hidden layer
        self.hidden_1 = torch.nn.Linear(11, 5)
        self.hidden_mid = torch.nn.Linear(5, 5)
        self.output = torch.nn.Linear(5, 1)

        self.sigma = torch.randn(1, requires_grad=True)

        # Define the activation functions that will be used
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1) # dim=1 calculates softmax across cols

    ## What should the output of the Neural Network be ............
    def forward(self, xi, xj):
        si = self.network_forward(xi)
        sj = self.network_forward(xj)

        s = torch.nn.Sigmoid()
        return s(si-sj*self.sigma)

    def network_forward(self, x):
        x = x.float()
        # Take input
        x = self.hidden_1(x)   # Linear Combination of input-> hidden
        x = self.sigmoid(x)    # Activation Function
        x = self.hidden_mid(x) # Linear Combination of input-> hidden
        x = self.sigmoid(x)    # Activation Function
        x = self.output(x)     # Linear Combination of hidden -> output
        x = self.sigmoid(x)    # Activation Function
        x = torch.flatten(x, start_dim=0, end_dim=-1)

        return x.double()

    ## How to Train the Model .....................................
    def train_model(self, lr, loss_fn, X, y, batch_size = 500):
        self.losses = []
        print('{0:10s} \t {1:10s}  {2:10s}'.format("Prediction", "Actual", "Loss"))
        for t in range(int(3e3)):  # loop over the dataset multiple times
            # Pick a random pair of values
            def pair(): return random.sample(range(X.shape[0]), 2)
            pairs = [pair() for i in range(batch_size)]
            left  = torch.tensor([X.numpy()[pair[0]] for pair in pairs], requires_grad=True)
            right = torch.tensor([X.numpy()[pair[1]] for pair in pairs], requires_grad=True)

            # Calculate the target for each pair
            S_ij = []
            for pair in pairs:
                S_ij.append(1 if y[pair[0]] > y[pair[1]] else 0)
            S_ij = torch.tensor(S_ij, requires_grad=True, dtype = float)

            # Forward Pass; Calculate the Prediction
            P_ij = self.forward(left, right)

            # Measure the Loss
            loss = loss_fn(S_ij, P_ij)
            if t % 100 == 0:
                # print('{0:10f} {1:10f} {2:10f}'.format(P_ij.item(), S_ij, loss.item()))
                print(loss.item())

            self.losses.append(loss.item())

            # Backward Pass; Calculate the Gradients
            loss.backward()


            # Zero the Gradients
            self.zero_grad()
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter -= lr * parameter.grad # Must be updated using `-=`, doesn't work otherwise
                self.sigma    -= lr * self.sigma.grad


# |  \/  (_)___  ___| | __ _ ___ ___(_)/ _(_) ___ __ _| |_(_) ___  _ __
# | |\/| | / __|/ __| |/ _` / __/ __| | |_| |/ __/ _` | __| |/ _ \| '_ \
# | |  | | \__ \ (__| | (_| \__ \__ \ |  _| | (_| (_| | |_| | (_) | | | |
# |_|  |_|_|___/\___|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|

#  ____       _
# |  _ \ __ _| |_ ___
# | |_) / _` | __/ _ \
# |  _ < (_| | ||  __/
# |_| \_\__,_|\__\___|

# Measure and report the misclassification Rate
class misclassification_rate:
    """
    This contains the functions to measure and report misclassification rates
    """
    def __init__(self, X, X_test, y, y_test, model):
        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test
        self.model = model

    def measure(self, X, y):
        """
        Measure the misclassification rate of model given input matrix
        X and desired output classifications y
        """
        yhat = self.model(X)
        yhat = yhat.detach().numpy().reshape(-1) > 0.5
        y=np.array(y)
        print(np.average(y != yhat))

    def report(self):
        print('\n----------------------------\n')
        print("The Training Missclassification rate is:\n")
        self.measure(self.X, self.y)
        print('\n----------------------------\n')
        print("The Testing Missclassification rate is:\n")
        self.measure(self.X_test, self.y_test)


main()
