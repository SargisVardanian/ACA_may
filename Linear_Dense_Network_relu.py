from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

import pandas as pd

dataset = pd.read_csv("new_data.csv")

X = dataset.drop('In-hospital_death', axis=1)
y = dataset['In-hospital_death']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.ones(output_size)
        self.activation = activation


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            self.output = self.relu(self.output)
            print('self.output', self.output.shape)
        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu':
            grad_output = self.def_relu(self.output) * grad_output

            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)

            grad_weights = self.def_relu(grad_weights)
            grad_biases = self.def_relu(grad_biases)
            print('grad_output', grad_output.shape)
        else:
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)

        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def relu(self, layer):
        # print('np.maximum(0.0, layer)', np.maximum(0.0, layer))
        return np.maximum(0.0, layer)

    def def_relu(self, layer):
        return np.where(layer > 0, 1, np.finfo(float).eps)

        # return np.where(layer > 0, 1, 0)


class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def fit(self, X_train, y_train, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            y_pred = self.forward(X_train)
            grad_output = 2 * (y_pred - y_train) / len(X_train)
            self.backward(grad_output, learning_rate)

            loss = np.abs(np.sum(y_pred - y_train) / len(X_train))
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}')

    def loss(self, X_test, y_test):
        y_pred = self.forward(X_test)
        loss = np.abs(np.sum(y_pred - y_test) / len(y_test))
        return loss

    def save_model(self, file_path):
        model_params = {
            'layers': self.layers
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_params, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)
        model = DenseNetwork()
        model.layers = model_params['layers']
        return model


dense_net = DenseNetwork()
input_dim = X_train.shape[1]
dense_net.add_layer(DenseLayer(input_dim, 10, activation='relu'))
dense_net.add_layer(DenseLayer(10, 5, activation='relu'))
dense_net.add_layer(DenseLayer(5, 1, activation='relu'))

learning_rate = 0.0008
num_epochs = 10000

dense_net.fit(X_train, y_train, learning_rate, num_epochs)

dense_net.save_model('trained_model_CNN.pkl')

# loaded_model = D\enseNetwork.load_model('trained_model_CNN.pkl')
# y_pred_dense = loaded_model.forward(X_test)

y_pred_dense = dense_net.forward(X_test)

print("Mean Squared Error (DenseNetwork implemented from scratch):", dense_net.loss(X_test, y_test))
