import torch
import numpy as np
from torch import nn
from progress.bar import Bar
import matplotlib.pyplot as plt
import sys
import math as m
from itertools import tee
import random


class three_layer_ranknet_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dtype, dev):
        super(three_layer_ranknet_network, self).__init__()
        self.wi =  torch.nn.Parameter(torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True, device = dev))
        self.wo = torch.nn.Parameter(torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True, device = dev))
                                                                                      
        self.bi = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype, requires_grad=True, device = dev))
        self.bo = torch.nn.Parameter(torch.randn(output_size, dtype=dtype, requires_grad=True, device = dev))
                                                                                      
        self.σ = torch.nn.Parameter(torch.randn(1, dtype=dtype, requires_grad=True, device = dev))

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
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.wo).add(self.bo)
        x = torch.sigmoid(x)

        return x

    def loss_fn(self, xi, xj, y):
        y_pred = self.forward(xi, xj)
        loss=torch.mean(-y*torch.log(y_pred)-(1-y)*torch.log(1-y_pred))
        return loss

    def train(self, x, target, η=1e-2, iterations=4e2, batch_size=50):
        if batch_size > x.shape[0]:
            batch_size = x.shape[0]-1
            print("\nWARNING: Batch Size Greater than training data, Batch Size set to nrow(data)\n", file = sys.stderr)
          
        opt = torch.optim.Adagrad(self.parameters(), lr=η)
        

        self.trainedQ = True
        bar = Bar('Processing', max=iterations*m.comb(batch_size, 2))
        for t in range(int(iterations)):
            sublosses = []
            vals = list(range(len(x)-1)) 
            vals = random.sample(vals, batch_size)
            for pair in pairwise(vals):
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

                # Backwards Pass
                # First Zero the Gradients, otherwise they can't be overwritten
                opt.zero_grad()

                # Calculate the Gradients
                loss.backward()

                # Adjust the Weights
                opt.step()

                bar.next()

            self.losses.append(np.average(sublosses))
        bar.finish()
        self.threshold_train(x, target, plot = False)


from itertools import chain
from itertools import combinations
def pairwise(iterable):
    "pairwise([1,2,3, 4]) --> [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]"
    s = list(iterable)
    pair_iter = chain.from_iterable(combinations(s, r) for r in [2])
    return pair_iter
