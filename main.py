import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

""" dataset """
california = fetch_california_housing()
X_numpy, y_numpy = california.data, california.target
X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy,test_size=0.2,random_state=42)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


""" Hyperparameters """
learning_rate = 0.0001
epochs = 10


""" neural network layers """
W1 = np.random.rand(8, 8)
b1 = np.zeros(8)

W2 = np.random.rand(8, 1)
b2 = np.zeros(1)

def forward_pass(X, y):
    y1 = X @ W1 + b1
    a1 = relu(y1)

    y2 = a1.T @ W2 + b2
    a2 = y2

    return 0.5 * (a2 - y)**2, a2, y2, a1, y1    # quadratic cost

""" training loop """
for epoch in range(epochs):
    loss = 0
    for X, y in zip(X_train, y_train):
        """ forward pass """
        cost, a2, y2, a1, y1 = forward_pass(X, y)
        loss += cost

        """ backward pass """
        lg = (a2 - y)
        dW2 = np.outer(a1, lg)
        db2 = lg
        
        hg = lg @ W2.T * relu_derivative(y1)
        dW1 = np.outer(X, hg)
        db1 = hg
        
        
        """ updating gradients """
        W2 = W2 - dW2 * learning_rate
        b2 = b2 - db2 * learning_rate

        W1 = W1 - dW1 * learning_rate
        b1 = b1 - db1 * learning_rate

    print(f"Epoch {epoch + 1} | Loss: {(loss.item() / X_train.shape[0]):.3f}")



""" eval loop """
loss = 0
for X, y in zip(X_test, y_test):
    cost, a2, y2, a1, y1 = forward_pass(X, y)
    loss += cost

print(f"Test Loss: {(loss.item() / X_test.shape[0]):.3f}")
