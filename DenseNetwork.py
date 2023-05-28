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


class DenseLayer:
    def __init__(self, neurons=64,  lr=0.01, max_iter=1000):
        self.neurons = neurons
        self.weights = None
        self.previous_layer = None
        self.max_iter = max_iter
        self.lr = lr

    def forward(self, X, weights=None, flag=False):
        self.weights = np.random.randn(X.shape[1] + 1, self.neurons)
        print('self.weights', self.weights.shape)
        X= np.insert(X, 0, 1, axis=1)
        if flag:
            return X @ weights
        return X @ self.weights

    def backward(self, X, y, grad_outputs):
        grad = np.dot(grad_outputs, self.weights.T)
        grad_weights = np.dot(X.T, grad_outputs)
        self.weights -= self.lr * grad_weights
        return grad, self.weights


class DenseNetwork:
    def __init__(self, layer, max_iter=1000):
        self.layers = layer
        self.max_iter = max_iter

    def forward(self, inputs, weights=None, flag=False):
        outputs = inputs
        self.inp_ar = []
        for layer in self.layers:
            outputs = layer.forward(outputs)
            self.inp_ar.append(outputs)
        if flag:
            for layer in self.layers:
                outputs = layer.forward(outputs, weights, flag=True)
        return outputs

    def backward(self, X, y, grad_outputs):
        grad_inputs = grad_outputs
        for layer in reversed(self.layers):
            grad_inputs, weights = layer.backward(X, y, grad_inputs)
        return grad_inputs, weights

    def fit(self, X, y):
        weights = []
        y = y.reshape(-1, 1)  # Reshape y to match the shape of outputs
        input = self.forward(X)
        inp_ar = [input]
        for _ in range(self.max_iter):
            # for _ in self.layers:
            input = self.forward(input)
            print('input', input.shape)
            inp_ar = self.inp_ar
            # print('inp_ar', inp_ar[1])
            for i in reversed(self.layers):
                output = self.backward(X, y, inp_ar[i])
                if i == self.layers[-1]:
                    grad_outputs = output - y
                else:
                    grad_outputs -= output
                weights.append(self.backward(X, y, grad_outputs)[1])
            self.weights = weights
    def predict(self, X):
        return self.forward(X, self.weights, flag=True)



layer1 = DenseLayer(neurons=64)

layer2 = DenseLayer(neurons=4)

iris = load_iris()

X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

network = DenseNetwork([layer1,layer2])

# Training the dense layer
network.fit(X, y)

predictions = network.predict(X_test)
print("Predictions:")
print(predictions, y_test)
