#!/usr/bin/env python
import os, sys
os.chdir("/home/ryan/Studies/2020ResearchTraining/ranknet/scripts/neural_network/")

# * Import Packages
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn
import random
from ranknet.three_layer_nn_class import three_layer_nn

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


# Main
def main():
    print("Hello World")

# Write a program in C to print filename and words to STDOUT, that way I can feed that to this.
#     _                   __ 
#    (_)___  ____  __  __/ /_
#   / / __ \/ __ \/ / / / __/
#  / / / / / /_/ / /_/ / /_  
# /_/_/ /_/ .___/\__,_/\__/  
#        /_/                 


def read_data():
    '''
    Read in all the variables into lists
    '''
    id_list      = []
    rating_list  = []
    words_list   = []
    for row in sys.stdin:
        id, rating, words = row.split("\t")
        id_list     = id_list + [id]
        rating_list = rating_list + [rating]
        words_list  = words_list + [words]
    
    return id_list, rating_list, words_list


#   ____________    ________  ______
#  /_  __/ ____/   /  _/ __ \/ ____/
#   / / / /_______ / // / / / /_    
#  / / / __/_____// // /_/ / __/    
# /_/ /_/       /___/_____/_/       
#                                   

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_matrix(corpus):
    '''
    Take a corpus of words and return a TF-IDF weighted DTM matrix.

    Parameters:
        corpus (list): A list containing documents, each document should itself
                    be a list of words, e.g.:

                    [
                        ['dont', 'panic'],
                        ['time', 'is', 'an', 'illusion'],
                        ['much', 'same', 'way', 'bricks', 'dont']
                    ]
    Output: 

        featrues (list): A list of features, i.e. terms/words corresponding to columns

        X (np.array):    A matrix of TF-IDF weighted values,
                         rows correspond to ID, columns to features.
    '''
    vectorizer = TfidfVectorizer()
    X          = vectorizer.fit_transform(corpus) # This is a DTM (as opposed to tdm) according to docs
    X = X.toarray().tolist() # Return a Dense numpy array not a sparse Scipy array, then turn the numpy into a nested list
    features   = vectorizer.get_feature_names()  # inspect size with  print(X.shape)
    return features, X


main()