#!/usr/bin/env python

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn
import random
from ranknet.test_cuda import test_cuda
from ranknet.make_data import make_data

# Set Seeds
torch.manual_seed(1)
np.random.seed(1)

# Set Torch Parameters
dtype = torch.float
dev = test_cuda()

# Set personal flags
DEBUG = True


# Main Function

def main():
    X_train, X_test, y_train, y_test = make_data(
        n=100, create_plot=True, dtype=dtype, dev=dev)


if __name__ == "__main__":
    main()
