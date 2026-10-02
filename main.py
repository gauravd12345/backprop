import torch
import numpy as np
from sklearn.datasets import fetch_california_housing

# extracting dataset
california = fetch_california_housing()
X_numpy, y_numpy = california.data, california.target

def relu(x):
    np.maximum(0, x, out=x)

""" Hyperparameters """
learning_rate = 0.01
epochs = 1
samples = 100

""" neural network layers """
W1 = np.ones((8, 8))
b1 = np.ones((1, 8))

W2 = np.ones((8, 1))
b2 = np.ones((1, 1))

""" training loop """
for _ in range(epochs):
    avg_loss = 0
    for X, y in zip(X_numpy[:samples], y_numpy[:samples]):

        """ forward pass """
        y1 = X @ W1 + b1
        relu(y1)
        y2 = y1 @ W2 + b2
        relu(y2)

        loss = y2 - y
        avg_loss += loss

        

    print(f"Loss: {avg_loss / samples}")

