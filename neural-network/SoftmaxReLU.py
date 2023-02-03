import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from keras.activations import linear, relu, sigmoid
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# make random dataset from a seed data using scikit datasets module
seed_data = [[-5,2], [-2,-2], [1,2], [5,-2]]
x_train, y_train = make_blobs(n_samples=2000, centers=seed_data, cluster_std=1.0, random_state=30)

# Define model and layers
softmax_model = Sequential(
    [
      Dense(25, activation='relu', name='L1'),
      Dense(15, activation='relu', name='L2'),
      Dense(4, activation='linear', name='L3'),
    ]
)


# Compute loss
softmax_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(0.001),
)

softmax_model.fit(x_train, y_train, epochs=10)

predictions = softmax_model.predict(x_train)
print(f"max value of predictions: \n{np.max(predictions)}")

# convert prediction to softmax probabilities
sm_predictions = tf.nn.softmax(predictions).numpy()
print(f"max value of softmax predictions: \n {np.max(sm_predictions)}")