#!/usr/bin/env python3

# * First load the packages
import torch
import math

dtype = float
device = torch.device('cpu')

# * Now build some data
import numpy
import scipy.stats

x = torch.randn(2000, device = device, dtype = dtype, requires_grad=False)
y = 0.5*(torch.exp(x) - torch.exp(-x))
y = scipy.stats.norm.cdf(x)
y = torch.from_numpy(y)

# ** Plot the Data
from plotnine import *
import pandas as pd

data = {'Input': x,
        'Output': y
        }

df = pd.DataFrame(data, columns = ['Input', 'Output'])

import copy

df_plot = copy.copy(df[3:])
df_plot['Output'] = df_plot['Output'].astype(float).round(3)
df_plot

p = (
    ggplot(df_plot, aes(x = 'Input', y = 'Output')) +
        geom_point() +
        theme_bw() +
        labs(x = "x", y = "int(X); X ~ N(0, 1)") +
        ggtitle('Cumulative Normal Distribution')

)

# fig = p.draw()
# fig.show()

# * Define the function we are going to use
# we want to use yhat = 1/(1+e^-x)

# * Implement the Neural Network
# Use the `.apply` method to call the function, `.forward` can't be used because the
# ctx object can't be given to it in order to get the output.

# ** Choose an arbitrary value for sigma
sigma = torch.randn((), device=device, dtype=dtype, requires_grad=True)
print(sigma)
eta   = 1e-6

print('iterations', '\t', 'loss')
print('----------------\n')
learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = 1/(1+torch.exp(-x*sigma))

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, '\t', loss.item())


    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        sigma -= learning_rate * sigma.grad

        # Manually zero the gradients after updating weights
        sigma.grad = None

print('\n')
print(f'Result: y = 1/(1+torch.exp(-x * {round(sigma.item())}))')


#########################################3
# ** Start the Loop
# *** Forward Pass; Calculate y
y_pred = 1/(1+torch.exp(-x*sigma))

# *** Calculate the Loss
loss = (y_pred - y).pow(2).sum().item()
print(y_pred)

# *** Backward Pass; Calculate the gradients and store in `.grad`
loss.backward()
print(loss.item())
print(sigma.grad)

# *** Adjust the Weights
with torch.no_grad():
    sigma = sigma - eta * sigma.grad

    # *** Reset the Gradient to None
    sigma.grad = None

    print(sigma)

#############################################
#############################################
#############################################
