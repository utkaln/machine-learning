import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
np.set_printoptions(precision=2)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# make dataset using scikit
seed_data = [[-5,2],[-2,-2],[1,2],[5,-2]]
x_train, y_train = make_blobs(n_samples=100, centers=seed_data, cluster_std=1.0, random_state=30)

# print all the class names
print(f"unique classes : \n{np.unique(y_train)}")

tf.random.set_seed(1234)
model = Sequential(
    [
      Dense(2, activation = 'relu', name='L1'),
      Dense(4, activation='linear', name='L2')
    ]
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(0.01),
)

model.fit(x_train, y_train, epochs=200)