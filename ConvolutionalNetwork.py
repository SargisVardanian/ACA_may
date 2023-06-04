import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os
import cv2
import glob

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        print('DenseLayer', input_size, output_size)
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.ones(output_size)
        self.activation = activation
        # self.m_w = np.zeros_like(self.weights)
        # self.v_w = np.zeros_like(self.weights)
        # self.m_b = np.zeros_like(self.biases)
        # self.v_b = np.zeros_like(self.biases)
        # self.beta1 = 0.9
        # self.beta2 = 0.999
        # self.epsilon = 1e-8
        # self.t = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            self.output = self.relu(self.output)
        elif self.activation == 'sigmoid':
            self.output = self.sigmoid(self.output)
        print('DenseLayer')
        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu':
            grad_output = self.def_relu(self.output) * grad_output
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)
            grad_weights = self.def_relu(grad_weights)
            grad_biases = self.def_relu(grad_biases)
        elif self.activation == 'sigmoid':  # Добавленная проверка для softmax
            grad_output = self.def_sigmoid(self.output) * grad_output
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)
            grad_weights = self.def_sigmoid(grad_weights)
            grad_biases = self.def_sigmoid(grad_biases)
        else:
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)
        # self.t += 1
        # self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_weights
        # self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_weights ** 2)
        # m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        # v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        # self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        #
        # self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_biases
        # self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_biases ** 2)
        # m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        # v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        # self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        # grad_input = np.dot(grad_output, self.weights.T)
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input

    def relu(self, layer):
        return np.maximum(0.0, layer)

    def def_relu(self, layer):
        return np.where(layer > 0, 1, np.finfo(float).eps)

    def sigmoid(self, layer):  # Добавлен метод sigmoid
        return 1 / (1 + np.exp(-layer))

    def def_sigmoid(self, layer):  # Добавлен метод def_sigmoid
        return self.sigmoid(layer) * (1 - self.sigmoid(layer))

        # return np.where(layer > 0, 1, 0)

# class DenseNetwork:
#     def __init__(self):
#         self.layers = []
#
#     def add_layer(self, layer):
#         self.layers.append(layer)
#
#     def forward(self, inputs):
#         for layer in self.layers:
#             inputs = layer.forward(inputs)
#         return inputs
#
#     def backward(self, grad_output, learning_rate):
#         for layer in reversed(self.layers):
#             grad_output = layer.backward(grad_output, learning_rate)
#
#     def fit(self, X_train, y_train, learning_rate, num_epochs):
#         for epoch in range(num_epochs):
#             y_pred = self.forward(X_train)
#             grad_output = 2 * (y_pred - y_train) / len(X_train)
#             self.backward(grad_output, learning_rate)
#
#             loss = np.abs(np.sum(y_pred - y_train) / len(X_train))
#             print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}')
#
#     def loss(self, X_test, y_test):
#         y_pred = self.forward(X_test)
#         loss = np.abs(np.sum(y_pred - y_test) / len(y_test))
#         return loss
#
#     def save_model(self, file_path):
#         model_params = {
#             'layers': self.layers
#         }
#         with open(file_path, 'wb') as f:
#             pickle.dump(model_params, f)
#
#     @staticmethod
#     def load_model(file_path):
#         with open(file_path, 'rb') as f:
#             model_params = pickle.load(f)
#         model = DenseNetwork()
#         model.layers = model_params['layers']
#         return model

class FlattenLayer:
    def __init__(self):
        self.input_shape = None
        self.flattened_size = None

    def forward(self, inputs):

        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        self.flattened_size = np.prod(inputs.shape[1:])
        flattened_inputs = inputs.reshape(batch_size, self.flattened_size)
        print('FlattenLayer', inputs.shape, flattened_inputs.shape)
        return flattened_inputs

    def backward(self, grad_output, learning_rate):
        grad_input = grad_output.reshape(self.input_shape)
        return grad_input


class ConvolutionalLayer:
    def __init__(self, input_shape, num_filters, filter_size, stride=1, activation='relu'):
        self.input_shape = input_shape
        print('ConvolutionalLayer\nself.input_shape', self.input_shape)
        self.stride = stride
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.weights = np.random.randn(filter_size, filter_size, input_shape[2], num_filters)
        # print('self.weights', self.weights.shape)
        self.biases = np.ones(num_filters)
        # self.m_w = np.zeros_like(self.weights)
        # self.v_w = np.zeros_like(self.weights)
        # self.m_b = np.zeros_like(self.biases)
        # self.v_b = np.zeros_like(self.biases)
        # self.beta1 = 0.9
        # self.beta2 = 0.999
        # self.epsilon = 1e-8
        # self.t = 0

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        output_height = (input_height - self.filter_size) // self.stride + 1
        output_width = (input_width - self.filter_size) // self.stride + 1
        self.output = np.zeros((batch_size, output_height, output_width, self.num_filters), dtype=np.float64)
        # print('self.output', self.output.shape)
        for i in range(output_height):
            for j in range(output_width):
                input_slice = inputs[:, i * self.stride:i * self.stride + self.filter_size,
                              j * self.stride:j * self.stride + self.filter_size, :, np.newaxis]
                self.output[:, i, j, :] = np.sum(input_slice * self.weights.transpose(0, 1, 2, 3),
                                                 axis=(1, 2, 3)) + self.biases
        # print('MSU,self.output', np.linalg.norm(self.output))
        if self.activation == 'relu':
            self.output = self.relu(self.output)
            # print('relu,self.output', np.linalg.norm(self.output))
        print('ConvolutionalLayer')
        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu':
            grad_output = self.def_relu(self.output) * grad_output
            # grad_biases = np.sum(grad_output, axis=0)

        batch_size, input_height, input_width, input_channels = self.inputs.shape
        _, output_height, output_width, _ = grad_output.shape
        grad_input = np.zeros_like(self.inputs)
        grad_weights = np.zeros_like(self.weights)

        for i in range(output_height):
            for j in range(output_width):
                input_slice = self.inputs[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for k in range(self.num_filters):
                    grad_input[:, i:i+self.filter_size, j:j+self.filter_size, :] += grad_output[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis] * self.weights[:, :, :, k]
                    grad_weights[:, :, :, k] += np.sum(input_slice * grad_output[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis], axis=0)

        # self.t += 1
        # self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_weights
        # self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_weights ** 2)
        # m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        # v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        # self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        #
        #
        # self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_biases
        # self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_biases ** 2)
        # m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        # v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        # self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        #

        grad_biases = np.sum(grad_output, axis=(0, 1, 2))
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        # print('np.linalg.norm(self.weights)', np.linalg.norm(self.weights))
        return grad_input

    def relu(self, layer):
        return np.maximum(0.0, layer)

    def def_relu(self, layer):
        return np.where(layer > 0, 1, np.finfo(float).eps)


class ConvolutionalNetwork:
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

    def fit(self, X_train, y_train, learning_rate, num_epochs, batch_size):
        for epoch in range(num_epochs):
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = batch_start + batch_size
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                print('X_batch', X_batch.shape, y_batch.shape)
                y_pred = self.forward(X_batch)
                # y_batch = y_batch.reshape(y_pred.shape)

                grad_output = 2 * (y_pred - y_batch) / len(X_batch)
                self.backward(grad_output, learning_rate)
                loss = np.abs(np.sum(y_pred - y_batch))
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_start // batch_size + 1}/{len(X_train) // batch_size + 1}, Loss: {loss}')
                if loss <= 3:
                    break
            if loss <= 3:
                break
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
        model = ConvolutionalNetwork()
        model.layers = model_params['layers']
        return model

class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        print("MaxPoolingLayer")
        self.pool_size = pool_size
        self.stride = stride
        self.inputs = None  # Добавленный атрибут inputs

    def forward(self, inputs):
        self.inputs = inputs  # Установка значения атрибута inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output_channels = input_channels
        output_shape = (batch_size, output_height, output_width, output_channels)
        outputs = np.zeros(output_shape)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                inputs_slice = inputs[:, h_start:h_end, w_start:w_end, :]
                outputs[:, i, j, :] = np.amax(inputs_slice, axis=(1, 2))
        print('MaxPoolingLayer')
        return outputs

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, output_channels = grad_output.shape
        grad_inputs = np.zeros_like(self.inputs)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                inputs_slice = self.inputs[:, h_start:h_end, w_start:w_end, :]
                max_values = np.amax(inputs_slice, axis=(1, 2), keepdims=True)
                mask = (inputs_slice == max_values)
                grad_inputs[:, h_start:h_end, w_start:w_end, :] += mask * grad_output[:, i:i+1, j:j+1, :]

        return grad_inputs


train_path = 'animals_dataset'

print('Specify the dimensions of the input images')
input_height = 224
input_width = 224
input_channels = 3

print('Load and preprocess the dataset')
class_names = os.listdir(train_path)
num_classes = len(class_names)

X_train = []
y_train = []
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(train_path, class_name)
    image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.resize(image, (input_height, input_width))
        X_train.append(image)
        y_train.append(class_index)

X_train = np.array(X_train)
y_train = np.array(y_train)

from keras.utils import to_categorical

print('Преобразование y_train в двоичное представление')
y_train = to_categorical(y_train)

print('Split the data into training and testing sets')
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

print('Shuffle the training data')
shuffle_indices = np.arange(X_train.shape[0])
np.random.shuffle(shuffle_indices)
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

print('Normalize the pixel values')
X_train = X_train / 255.0
X_test = X_test / 255.0



print('Reshape the input data')
X_train = X_train.reshape((-1, input_height, input_width, input_channels))
X_test = X_test.reshape((-1, input_height, input_width, input_channels))


print('Create and train the convolutional network')
conv_net = ConvolutionalNetwork()
batch_size = 30
X_batch = X_train[:batch_size]
y_batch = y_train[:batch_size]
print('y_batch', y_batch)

input_shape = X_batch.shape

conv_net.add_layer(ConvolutionalLayer((224, 224, 3), num_filters=10, filter_size=3, stride=1, activation='relu'))
conv_net.add_layer(ConvolutionalLayer((222, 222, 10), num_filters=10, filter_size=3, stride=1, activation='relu'))
conv_net.add_layer(MaxPoolingLayer(pool_size=2, stride=2))
conv_net.add_layer(ConvolutionalLayer((111, 111, 10), num_filters=5, filter_size=3, stride=1, activation='relu'))
conv_net.add_layer(MaxPoolingLayer(pool_size=2, stride=2))
conv_net.add_layer(FlattenLayer())
conv_net.add_layer(DenseLayer((54 * 54 * 5), 256, activation='relu'))
conv_net.add_layer(DenseLayer(256, num_classes-1, activation='sigmoid'))

learning_rate = 0.0008
num_epochs = 5

print('conv_net.fit')
# conv_net.fit(X_train, y_train, learning_rate, num_epochs, batch_size=batch_size)
# conv_net.save_model('trained_model_CNN.pkl')


print('conv_net.predict')

loaded_model = conv_net.load_model('trained_model_CNN.pkl')
# y_pred_dense = loaded_model.forward(X_test)


# y_pred_conv = conv_net.forward(X_test)
# print("Mean Squared Error (Convolutional Network implemented from scratch):", conv_net.loss(X_train, y_train))

print("Mean Squared Error (Convolutional Network implemented from scratch):", loaded_model.loss(X_train, y_train))

