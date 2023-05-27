import numpy as np
from sklearn.datasets import load_iris
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


iris = load_iris()

# X = iris.data
# y = iris.target

X, y = make_regression(n_samples=100, n_informative=1, n_features=3, n_targets=1)

print(X)

ner = Neuron()

r = ner.call(X, y, training=True)