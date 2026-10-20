# Manual Backpropagation with NumPy

A from-scratch feedforward neural network trained on the California Housing dataset using NumPy. The project implements the forward pass, quadratic loss, backpropagation, and stochastic gradient descent without using a deep-learning framework.

## Overview

The network predicts the median house value from the eight numerical features in the California Housing dataset.

```text
Input:  8 features
   ↓
Dense layer: 8 neurons + ReLU
   ↓
Dense output layer: 1 neuron + linear activation
   ↓
Predicted house value
```

The implementation is intended as an educational example of how neural-network training works internally.

## Requirements

- Python 3.9+
- NumPy
- scikit-learn

Install the dependencies with:

```bash
pip install numpy scikit-learn
```

## Running the project

Save the training script as a Python file, such as `manual_backprop.py`, and run:

```bash
python manual_backprop.py
```

The script prints the average training loss after each epoch and the final average test loss.

## Dataset

The project uses `fetch_california_housing()` from scikit-learn. The dataset contains eight input features describing California districts and a continuous target representing median house value.

The data is split into:

- 80% training data
- 20% test data

The split uses `random_state=42` for reproducibility of the dataset split.

## Forward pass

For one input example \(x\), the first layer computes:

\[
z_1 = xW_1 + b_1
\]

The ReLU activation is then applied:

\[
a_1 = \operatorname{ReLU}(z_1)
\]

The output layer is linear:

\[
z_2 = a_1W_2 + b_2
\]

Since this is a regression problem, the output is not passed through an activation function:

\[
\hat{y} = z_2
\]

The per-example quadratic loss is:

\[
L = \frac{1}{2}(\hat{y} - y)^2
\]

## Manual backpropagation

The output error is:

\[
\frac{\partial L}{\partial z_2} = \hat{y} - y
\]

The gradients for the output layer are:

\[
\frac{\partial L}{\partial W_2} = a_1^T\frac{\partial L}{\partial z_2}
\]

\[
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}
\]

The error propagated into the hidden layer is:

\[
\frac{\partial L}{\partial z_1}
=
\left(\frac{\partial L}{\partial z_2}W_2^T\right)
\odot \operatorname{ReLU}'(z_1)
\]

The gradients for the first layer are:

\[
\frac{\partial L}{\partial W_1} = x^T\frac{\partial L}{\partial z_1}
\]

\[
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}
\]

Each parameter is updated using gradient descent:

\[
W \leftarrow W - \eta\frac{\partial L}{\partial W}
\]

where \(\eta\) is the learning rate.

## Hyperparameters

```python
learning_rate = 0.0001
epochs = 10
```

Training uses one sample at a time, so the optimization method is stochastic gradient descent rather than full-batch gradient descent.

## Important implementation notes

- The weights are initialized randomly and are not currently given a fixed NumPy seed. Results may therefore vary between runs.
- The input features are not normalized. Feature scaling can make training more stable and may allow a larger learning rate.
- Ten epochs is a small demonstration run; increasing the number of epochs may improve performance.
- The loss printed by the script is the average quadratic loss over the samples in the split.
- The code uses `np.outer` to construct the weight gradients for each individual training example.
- The test loop evaluates the learned parameters without updating them.

## Learning goals

This project demonstrates:

1. How a dense neural network performs a forward pass.
2. How the chain rule produces gradients layer by layer.
3. How ReLU affects gradient flow.
4. How parameters are updated manually with gradient descent.
5. How a regression model can be evaluated on unseen data.

## Possible improvements

- Normalize the input features.
- Add a fixed random seed for reproducible weight initialization.
- Increase the number of epochs.
- Track and plot training and test loss.
- Use mini-batch training instead of one-sample-at-a-time updates.
- Add validation data and early stopping.
- Compare the manual implementation with a PyTorch or TensorFlow version.

## License

This project is intended for educational use. Add a license appropriate for your repository if you plan to distribute or reuse the code.
