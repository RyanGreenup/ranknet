import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def make_data(create_plot=False, n=1000, dtype=torch.float, dev="cpu"):
    # X, y = datasets.make_blobs(n, 2, 2, random_state=7)
    X, y = datasets.make_moons(n_samples=n, noise=0.1, random_state=0) # Moons Data for later

    # Reshape the data to be consistent
    y = np.reshape(y, (len(y), 1))  # Make y vertical n x 1 matrix.

    # -- Split data into Training and Test Sets --------------------
    data = train_test_split(X, y, test_size=0.4)

    if(create_plot):
        # Create the Scatter Plot
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Sample Data")
        plt.show()

    # Make sure we're working with tensors not mere numpy arrays
    torch_data = [None]*len(data)
    for i in range(len(data)):
        torch_data[i] = torch.tensor(data[i], dtype=dtype, requires_grad=False)

    return torch_data
