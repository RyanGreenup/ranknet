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
class sigmoid(torch.autograd.Function):
    """
    The Sigmoid Curve, this will calculate the output and contain the partial derivative
    """
    @staticmethod
    def forward(ctx, sigma, input):
        """
        The forward pass receives an input tensor and outputs the
        value of the sigmoid function applied to each element.

        ctx is a context object that can be used to cache arbitrary objects for
        use in the backward pass (e.g. maybe to save a variable like sqrt() which is
        computationally expensive and could be used in the backward pass)

        additional parameters can also be defined, but, input must be last.
        """
        ctx.save_for_backward(input, sigma)
        return 1/(1+torch.exp(-input*sigma))

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass returns the partial derivative of the function with
        respect to the Error.

        To do this calculate ∂f/∂x then return:

            grad_output*∂f/∂x

        because grad_output will be equal to ∂E/∂f the chain rule will cancel it out.

        but each input must have a gradient returned.
        """
        input, sigma = ctx.saved_tensors                             # The comma unpacks the list item
        grad_sigma = x*torch.exp(-sigma*x)/((1+torch.exp(-sigma*x)).pow(2))
        grad_input = torch.exp(-input)/((1+torch.exp(-input)).pow(2))

        #
        return grad_sigma, grad_input


# * Implement the Neural Network
# Use the `.apply` method to call the function, `.forward` can't be used because the
# ctx object can't be given to it in order to get the output.

# ** Choose an arbitrary value for sigma
sigma = torch.randn((), device=device, dtype=dtype, requires_grad=True)
print(sigma)
eta   = 1e-6

# ** Start the Loop
# *** Forward Pass; Calculate y
y_pred = sigmoid.apply(x, sigma)

# *** Calculate the Loss
loss = (y_pred - y).pow(2).sum()

# *** Backward Pass; Calculate the gradients and store in `.grad`
loss.backward()
print(loss.item())

# *** Adjust the Weights
sigma = sigma - eta * sigma.grad

print(sigma)

#############################################
#############################################
#############################################
