import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("new_data.csv")

X = dataset.drop('In-hospital_death', axis=1)
y = dataset['In-hospital_death']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print('dataset', dataset.shape)

model.fit(X_train, y_train, epochs=30, batch_size=2, validation_data=(X_test, y_test))

model.save("model.h5")

y_pred_prob = model.predict(X_test)

threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)