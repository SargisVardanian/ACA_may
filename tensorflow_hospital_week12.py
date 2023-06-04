import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("new_data.csv")

X = dataset.drop('In-hospital_death', axis=1)
y = dataset['In-hospital_death']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


class DenseLayer(tf.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        self.W = tf.Variable(tf.random.normal([input_dim, output_dim], stddev=0.1))
        self.b = tf.Variable(tf.random.normal([output_dim], stddev=0.1))
        print("self.W, self.b0", self.W.shape, self.b.shape)

    def __call__(self, x):
        print("self.W, self.b1", self.W.shape, self.b.shape)
        return tf.nn.relu(tf.matmul(x, self.W) + self.b)

class SurvivalModel(tf.Module):
    def __init__(self):
        super(SurvivalModel, self).__init__()
        self.layers = []

    def add_layer(self, input_dim, output_dim):
        self.layers.append(DenseLayer(input_dim, output_dim))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @tf.function
    def train_step(self, inputs, targets, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            predictions = self(inputs)
            loss = loss_fn(targets, predictions)
        if tf.math.is_nan(loss):
            return 0.0
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self, X_train, y_train, num_epochs, batch_size, optimizer, loss_fn):
        num_batches = len(X_train) // batch_size
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                batch_inputs = tf.convert_to_tensor(X_train[start:end], dtype=tf.float32)
                batch_targets = tf.convert_to_tensor(y_train[start:end], dtype=tf.float32)
                # print('batch_inputs,batch_targets', batch_inputs.shape, batch_targets)
                loss = self.train_step(batch_inputs, batch_targets, optimizer, loss_fn)
                # print('loss', loss.numpy())
                epoch_loss += loss
            if (epoch + 1) % 100 == 0:
                print('Epoch {}/{} - Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss / num_batches))

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
        model = SurvivalModel()
        model.layers = model_params['layers']
        return model


model = SurvivalModel()
input_dim = X_train.shape[1]  # Get the number of input features
model.add_layer(input_dim=input_dim, output_dim=16)
model.add_layer(input_dim=16, output_dim=9)
model.add_layer(input_dim=9, output_dim=9)
model.add_layer(input_dim=9, output_dim=1)
optimizer = tf.optimizers.Adam(learning_rate=0.0001)


print('DenseLayer', DenseLayer(input_dim=input_dim, output_dim=1))

num_epochs = 1200
batch_size = 40
model.train(X_train, y_train, num_epochs, batch_size, optimizer, model.loss_fn)

model.save_model("saved_model_week12.pkl")
# model = SurvivalModel.load_model("saved_model_week12.pkl")


X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
predictions = model(X_test)

print('Survival probability:', model.loss_fn(y_test, predictions))
