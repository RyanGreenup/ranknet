import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import tee
from torch import nn
from progress.bar import Bar

class three_layer_nn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dtype=torch.float, dev="cpu"):
        super(three_layer_nn, self).__init__()
        self.wi = torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True, device = dev)
        self.wo = torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True, device = dev)
                                                                                      
        self.bi = torch.randn(hidden_size, dtype=dtype, requires_grad=True, device = dev)
        self.bo = torch.randn(output_size, dtype=dtype, requires_grad=True, device = dev)
                                                                                      
        self.σ = torch.randn(1, dtype=dtype, requires_grad=True, device = dev)

        self.losses = []  # List to hold loss after each iteration of training
        self.trainedQ = False  # Has the model been trained yet?

    def threshold_train(self, X, y, plot = False):
        self.threshold = 0.5                   # Threshold used for Misclassification Rate
        rate_best = 1                          # Set default misclassification as totally wrong.

        t_list = [i/100 for i in range(100)]  # threshold to try
        rates = []                            # List of misclassification rates
        for t in t_list:
            rate = self.misclassification_rate(X, y, threshold = t)
            rates.append(rate)
            if (rate < rate_best):
                rate_best = rate
                self.threshold = t

        if plot:    
            plt.plot(t_list, rates)
            plt.title('Misclassification Rate Given Thresholds')
            plt.show()


    def misclassification_rate(self, X, target, threshold):
        if not self.trainedQ:
            sys.stderr("WARNING: Model has not yet been trained, use the train method")
        rates = []
        for pair in pairwise(range(len(X)-1)):
            xi =      X[pair[0],]
            yi = target[pair[0]]
            xj =      X[pair[1],]
            yj = target[pair[1]]

            # rencode from {0, 1} to {-1, 0, 1}
            y = ((yi > yj)*2 - 1)*(yi != yj)
            y = y.item()

            # Calculate y, forward pass
            y_pred = self.forward(xi, xj).item()
            if (y_pred < -threshold):
                y_pred = -1
            elif (y_pred > threshold):
                y_pred = 1
            else: 
                y_pred = 0
            m_rate = int(y_pred != y)


            rates.append(m_rate)
        return np.average(rates)
    

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
        loss=torch.mean(-y*torch.log(y_pred)-(1-y)*torch.log(1-y_pred))
        return loss

    def train(self, x, target, η=1e-2, iterations=4e2):
        self.trainedQ = True
        bar = Bar('Processing', max=iterations)
        for t in range(int(iterations)):
            sublosses = []
            for pair in pairwise(range(len(x)-1)):
                xi =      x[pair[0],]
                yi = target[pair[0]]
                xj =      x[pair[1],]
                yj = target[pair[1]]

                # encode from {0, 1} to {-1, 0, 1}
                y = yi-yj
                # Scale between {0,1}
                y = 1/2*(1+y)
                

                # Calculate y, forward pass
                y_pred = self.forward(xi, xj)

                # Measure the loss
                loss = self.loss_fn(xi, xj, y)
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
        self.threshold_train(x, target, plot = True)


def pairwise(iterable): # NOTE https://docs.python.org/3/library/itertools.html
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
