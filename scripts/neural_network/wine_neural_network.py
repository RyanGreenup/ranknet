#!/usr/bin/env python

#    _  _     ___                            _   
#  _| || |_  |_ _|_ __ ___  _ __   ___  _ __| |_ 
# |_  ..  _|  | || '_ ` _ \| '_ \ / _ \| '__| __|
# |_      _|  | || | | | | | |_) | (_) | |  | |_ 
#   |_||_|   |___|_| |_| |_| .__/ \___/|_|   \__|
#                          |_|                   
#  ____            _                         
# |  _ \ __ _  ___| | ____ _  __ _  ___  ___ 
# | |_) / _` |/ __| |/ / _` |/ _` |/ _ \/ __|
# |  __/ (_| | (__|   < (_| | (_| |  __/\__ \
# |_|   \__,_|\___|_|\_\__,_|\__, |\___||___/
#                            |___/           

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


#  _                    _   ____        _        
# | |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
# | |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
# | |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
# |_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
                                               
df=pd.read_csv('./DataSets/winequality-red.csv', sep=';')
df=np.array(df.values)

# Extract the Features
y = df[:,-1]
X = df[:,range(df.shape[1]-1)]

# Make the data categorical
y = y>5

# Transfom the Data into Tensors
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))



# |  \/  (_)___  ___| | __ _ ___ ___(_)/ _(_) ___ __ _| |_(_) ___  _ __  
# | |\/| | / __|/ __| |/ _` / __/ __| | |_| |/ __/ _` | __| |/ _ \| '_ \ 
# | |  | | \__ \ (__| | (_| \__ \__ \ |  _| | (_| (_| | |_| | (_) | | | |
# |_|  |_|_|___/\___|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|
                                                                       
#  ____       _       
# |  _ \ __ _| |_ ___ 
# | |_) / _` | __/ _ \
# |  _ < (_| | ||  __/
# |_| \_\__,_|\__\___|
                    

def misclassification_rate(X, y):
    yhat = net(X)
    yhat = yhat.detach().numpy().reshape(-1) > 0.3

    y=np.array(y)

    print(np.average(y != yhat))



#  _   _                      _   _   _      _                      _    
# | \ | | ___ _   _ _ __ __ _| | | \ | | ___| |___      _____  _ __| | __
# |  \| |/ _ \ | | | '__/ _` | | |  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ /
# | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   < 
# |_| \_|\___|\__,_|_|  \__,_|_| |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\
                                                                       

# Define the Class for Torch

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        # Create the input layer and the hidden layer
        self.hidden_1 = nn.Linear(11, 5)
        self.hidden_mid = nn.Linear(5, 5)
        self.output = nn.Linear(5, 1)

        # Define the activation functions that will be used
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) # dim=1 calculates softmax across cols

    def forward(self, x):
        # Take input
        x = self.hidden_1(x)  # Linear Combination of input-> hidden
        x = self.sigmoid(x) # Activation Function
        x = self.hidden_mid(x)  # Linear Combination of input-> hidden
        x = self.sigmoid(x) # Activation Function
        x = self.hidden_mid(x)  # Linear Combination of input-> hidden
        x = self.sigmoid(x) # Activation Function
        x = self.output(x)  # Linear Combination of hidden -> output
#        x = self.sigmoid(x) # Activation Function
        x = torch.flatten(x, start_dim=0, end_dim=-1)

        return x

# Assign the model object

net = Network()
print('The Neural Network is described as:\n')
print(net)


## Print the Model Output
print('The current output of the neural network with random weights are:')
out = net(X)
print(out)




#  _                    _____                 _   _             
# | |    ___  ___ ___  |  ___|   _ _ __   ___| |_(_) ___  _ __  
# | |   / _ \/ __/ __| | |_ | | | | '_ \ / __| __| |/ _ \| '_ \ 
# | |__| (_) \__ \__ \ |  _|| |_| | | | | (__| |_| | (_) | | | |
# |_____\___/|___/___/ |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|
                                                              
eta = 1/10

import torch.optim as optim

loss_fn = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=eta, momentum = 0.9)

# loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(net.parameters(), lr = eta)

## Print THe Misclassication rate
loss_fn(net(X), y.reshape((-1, 1))).item()


# |_   _| __ __ _(_)_ __   | |_| |__   ___  |  \/  | ___   __| | ___| |
#   | || '__/ _` | | '_ \  | __| '_ \ / _ \ | |\/| |/ _ \ / _` |/ _ \ |
#   | || | | (_| | | | | | | |_| | | |  __/ | |  | | (_) | (_| |  __/ |
#   |_||_|  \__,_|_|_| |_|  \__|_| |_|\___| |_|  |_|\___/ \__,_|\___|_|
                                                                     
for t in range(10000):  # loop over the dataset multiple times
    # Forward Pass; Calculate the Prediction
    y_pred = net(X)

    # Zero the Gradients
    optimizer.zero_grad()

    # Measure the Loss
    loss = loss_fn(y, y_pred)
    if t % 100 == 0:
        print(loss.item())

    # Backward Pass; Calculate the Gradients
    loss.backward()

    # update the Weights
    optimizer.step()

loss_fn(net(X), y.reshape((-1, 1))).item()
misclassification_rate(X, y)
