import torch
import numpy as np
from torch import nn
from progress.bar import Bar


class three_layer_classification_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dtype=torch.float, dev="cpu"):
        super(three_layer_classification_network, self).__init__()
        self.wi = torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True)
        self.wo = torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True)

        self.bi = torch.randn(hidden_size, dtype=dtype, requires_grad=True)
        self.bo = torch.randn(output_size, dtype=dtype, requires_grad=True)
        self.σ = torch.randn(output_size, dtype=dtype, requires_grad=True)

        self.losses = []

    def forward(self, x):
        x = torch.matmul(x, self.wi).add(self.bi)
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.wo).add(self.bo)
        x = torch.sigmoid(self.σ*x) # setting up sigma now is convenient

        return x

    def loss_fn(self, x, y):
        y_pred = self.forward(x)
        return torch.mean(torch.pow((y-y_pred), 2))

    def misclassification_rate(self, x, y):
        y_pred = (self.forward(x) > 0.5)
        return np.average(y != y_pred)

    def train(self, x, target, η=30, iterations=2e4):
        bar = Bar('Processing', max=iterations) # progress bar
        for t in range(int(iterations)):

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
                self.σ  -= η * self.σ.grad;  self.σ.grad = None
            bar.next()
        bar.finish()
                # ; Zero out the gradients, they've been used
