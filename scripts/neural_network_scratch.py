#!/usr/bin/env python

#!/usr/bin/env python

#    _  _     ___                            _   
#  _| || |_  |_ _|_ __ ___  _ __   ___  _ __| |_ 
# |_  ..  _|  | || '_ ` _ \| '_ \ / _ \| '__| __|
# |_      _|  | || | | | | | |_) | (_) | |  | |_ 
#   |_||_|   |___|_| |_| |_| .__/ \___/|_|   \__|
#                          |_|                   
#  ____            _                         
# |  _ \ __ _  ___| | ____ _  __ _  ___  ___ 
# | |_) / _` |/ __| |/ / _` |/ _` |/ _ \/ __|
# |  __/ (_| | (__|   < (_| | (_| |  __/\__ \
# |_|   \__,_|\___|_|\_\__,_|\__, |\___||___/
#                            |___/           

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
print()

#  _                    _   ____        _        
# | |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
# | |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
# | |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
# |_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
                                               
df=pd.read_csv('./DataSets/winequality-red.csv', sep=';')
df=np.array(df.values)

# Extract the Features
y = df[:,-1]
X = df[:,range(df.shape[1]-1)]

# Make the data categorical
y = y>5


# |  \/  (_)___  ___| | __ _ ___ ___(_)/ _(_) ___ __ _| |_(_) ___  _ __  
# | |\/| | / __|/ __| |/ _` / __/ __| | |_| |/ __/ _` | __| |/ _ \| '_ \ 
# | |  | | \__ \ (__| | (_| \__ \__ \ |  _| | (_| (_| | |_| | (_) | | | |
# |_|  |_|_|___/\___|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|
                                                                       
#  ____       _       
# |  _ \ __ _| |_ ___ 
# | |_) / _` | __/ _ \
# |  _ < (_| | ||  __/
# |_| \_\__,_|\__\___|
                    

def misclassification_rate(X, y):
    yhat = net(X)
    yhat = yhat.detach().numpy().reshape(-1) > 0.3

    y=np.array(y)

    print(np.average(y != yhat))

def loss(X, y)
    yhat = net(X)
    y
    return sum((y-yhat)**2)

#  _   _                      _   _   _      _                      _    
# | \ | | ___ _   _ _ __ __ _| | | \ | | ___| |___      _____  _ __| | __
# |  \| |/ _ \ | | | '__/ _` | | |  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ /
# | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   < 
# |_| \_|\___|\__,_|_|  \__,_|_| |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\
                                                                       
# 11-4-1 Network
def net(X, w1, w2):
    net1     = np.matmul(w1, X)
    hidden1  = step(net1)
    net2     = np.matmul(w2, hidden1) 
    out      = step(net2)

# Calculate the Gradients
# I need to be more careful with my derivatives of matrices though.
def backward(X, y, w1, w2):
    pe_pout = np.matmul
