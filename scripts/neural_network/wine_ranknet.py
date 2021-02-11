#!/usr/bin/env python
import os, sys
os.chdir("/home/ryan/Studies/2020ResearchTraining/ranknet/scripts/neural_network/")

# * Import Packages
#   ___                            _
#  |_ _|_ __ ___  _ __   ___  _ __| |_
#   | || '_ ` _ \| '_ \ / _ \| '__| __|
#   | || | | | | | |_) | (_) | |  | |_
#  |___|_| |_| |_| .__/ \___/|_|   \__|
#                |_|
from sklearn import datasets
import numpy as np
from progress.bar import Bar
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn
from itertools import tee
import random

torch.manual_seed(1)  # set the seed.
np.random.seed(1)

dtype = torch.float


if torch.cuda.is_available():
  print("Detected Cuda Cores, setting Device to Cuda")
  dev = "cuda:0"
else:
  print("No cuda cores detected, using CPU")
  dev = "cpu"
dev = "cpu"
DEBUG = True

# * Main
#  __  __       _
# |  \/  | __ _(_)_ __
# | |\/| |/ _` | | '_ \
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|
#

def main():
    # Load Data ................................................
    X_train, X_test, y_train, y_test = load_data()


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



# |  \/  (_)___  ___| | __ _ ___ ___(_)/ _(_) ___ __ _| |_(_) ___  _ __
# | |\/| | / __|/ __| |/ _` / __/ __| | |_| |/ __/ _` | __| |/ _ \| '_ \
# | |  | | \__ \ (__| | (_| \__ \__ \ |  _| | (_| (_| | |_| | (_) | | | |
# |_|  |_|_|___/\___|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|

#  ____       _
# |  _ \ __ _| |_ ___
# | |_) / _` | __/ _ \
# |  _ < (_| | ||  __/
# |_| \_\__,_|\__\___|

main()
