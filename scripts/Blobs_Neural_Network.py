#!/usr/bin/env python

# Import Packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn
import random
from ranknet.test_cuda import test_cuda

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
    print("Hello World!")
