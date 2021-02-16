import torch
import numpy as np
from torch import nn


class three_layer_classification_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dtype=torch.float, dev="cpu"):
        super(three_layer_classification_network, self).__init__()
        self.wi = torch.randn(input_size, hidden_size, dtype=dtype, requires_grad=True)
        self.wo = torch.randn(hidden_size, output_size, dtype=dtype, requires_grad=True)

        self.bi = torch.randn(hidden_size, dtype=dtype, requires_grad=True)
        self.bo = torch.randn(output_size, dtype=dtype, requires_grad=True)

        self.losses = []

    def forward(self, x):
        x = torch.matmul(x, self.wi).add(self.bi)
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.wo).add(self.bo)
        x = torch.sigmoid(x)
        return x

    def loss_fn(self, x, y):
        y_pred = self.forward(x)
        return torch.mean(torch.pow((y-y_pred), 2))

    def misclassification_rate(self, x, y):
        y_pred = (self.forward(x) > 0.5)
        return np.average(y != y_pred)