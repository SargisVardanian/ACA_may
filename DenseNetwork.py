import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression

class Neuron:
    def __init__(self, lr=0.1, max_iter=1000):
        self.max_iter = max_iter
        self.lr = lr

    def call(self, X, y=None, training=False):
        if training:
            X = np.insert(X, 0, 1, 1)
            self.w = np.random.rand(X.shape[1])
            for i in range(self.max_iter):
                grad = (-2 * X.T @ (y - (X @ self.w)))/ X.shape[0]
                self.w = self.w - self.lr * grad
                # print('gard', grad, np.linalg.norm(grad))
        else:
            X = np.insert(X, 0, 1, 1)
            y_pred = X @ self.w
            return y_pred


import numpy as np


class DenseLayer1:
    def __init__(self, neurons=64, lr=0.01, max_iter=1000):
        self.neurons = neurons
        self.weights1 = None
        self.weights2 = None
        self.biases = None
        self.lr = lr
        self.max_iter = max_iter

    def call(self, X, y=None, training=False):
        if training:
            X = np.insert(X, 0, 1, axis=1)
            if self.weights1 is None:
                self.weights1 = np.random.randn(X.shape[1], self.neurons)
                self.weights2 = np.random.randn(self.neurons, y.shape[1])
                self.biases = np.zeros((1, self.neurons))

            for _ in range(self.max_iter):
                hidden_output = np.dot(X, self.weights1) + self.biases
                output = np.dot(hidden_output, self.weights2)
                grad_output = -2 * (y - output) / X.shape[0]
                grad_hidden = np.dot(grad_output, self.weights2.T)

                self.weights2 -= self.lr * np.dot(hidden_output.T, grad_output)
                self.weights1 -= self.lr * np.dot(X.T, grad_hidden)
                self.biases -= self.lr * np.sum(grad_hidden, axis=0)

                print('np.linalg.norm(grad_hidden)', np.linalg.norm(grad_hidden))

                if np.linalg.norm(grad_hidden) < 0.1:
                    break
        else:
            X = np.insert(X, 0, 1, axis=1)
            hidden_output = np.dot(X, self.weights1) + self.biases
            output = np.dot(hidden_output, self.weights2)
            return output


class DenseLayer:
    def __init__(self, neurons=64):
        self.neurons = neurons
        self.weights = None
        self.biases = None

    def call(self, X, y=None, training=False):
        if training:
            X = np.insert(X, 0, 1, axis=1)
            if self.weights is None:
                self.weights = np.random.randn(X.shape[1], self.neurons)
                self.biases = np.zeros((1, self.neurons))

            for _ in range(self.max_iter):
                output = np.dot(X, self.weights) + self.biases
                grad = (-2 * X.T @ (output - y)) / X.shape[0]
                self.weights -= self.lr * grad
                self.biases -= self.lr * np.sum(grad, axis=0)
        else:
            X = np.insert(X, 0, 1, axis=1)
            output = np.dot(X, self.weights) + self.biases
            return output

layer = DenseLayer1(neurons=4, lr=0.1, max_iter=1000)

iris = load_iris()

X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Generating some random training data
np.random.seed(42)
X = np.random.randn(100, 4)
# print('inputs', X)
y = np.random.randn(100, 1)
# print('targets', y.shape[1])

# Training the dense layer
layer.call(X, y, training=True)

# Making predictions
predictions = layer.call(X_test)
print("Predictions:")
print(predictions, y_test)
