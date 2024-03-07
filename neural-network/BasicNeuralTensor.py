# Uses Tensorflow 2.0
# Uses Keras - Framework that helps create layer centric interface in Tensorflow
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.activations import sigmoid
from LoadLogRegData import loadData

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# Dataset
x_train = np.array([0,1,2,3,4,5]).reshape(-1,1)
y_train = np.array([0,0,0,1,1,1]).reshape(-1,1)

print(f"x_train: {x_train}\ny_train: {y_train}")

# Logistic Neuron using Tensorflow
model = Sequential(
    [
      Dense(activation= 'sigmoid', name= 'L1', units=1, input_dim=1)
    ]
)

model.summary()

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(f"w: {w}\nb: {b}")


# ==========================
# More advanced example of Tensorflow

#X = np.array([[185.32,  12.69], [259.92, 11.87], [231.01,  14.41], [175.37,  11.72], [187.12,  14.13],[225.91,  12.1 ],[208.41,  14.18], [207.08,  14.03],[280.6,   14.23], [202.87,  12.25]])
#Y = np.array([[1.],[0.], [0.], [0.], [1.], [1.],[0.],[0.],[0.],[1.]])
X, Y = loadData()
# Normalize Data
norm_1 = tf.keras.layers.Normalization(axis=-1)
norm_1.adapt(X)
Xn = norm_1(X)

# Tile/ copy data to increase training data size
Xt = np.tile(Xn, (1000,1))
Yt = np.tile(Y, (1000,1))

# Create Tensorflow model

# the following line is to make sure results are consistent
tf.random.set_seed(1234)

model = Sequential(
    [
      tf.keras.Input(shape=(2,)),
      Dense (3, activation='sigmoid', name= 'layer1'),
      Dense (1, activation='sigmoid', name= 'layer2')         
    ]
)

# Print description of the neural network
model.summary()

# Find out the initial weights (w, b) that model has assigned
w1, b1 = model.get_layer("layer1").get_weights()
w2, b2 = model.get_layer("layer2").get_weights()

print(f"Initial w1: {w1}\nb1: {b1}\nw2: {w2}\nb2: {b2}")

# Define loss function 
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
                                         
# Run gradient descent
model.fit(Xt, Yt, epochs=10,)

# Find updated weights after gradient descent
w1, b1 = model.get_layer("layer1").get_weights()
w2, b2 = model.get_layer("layer2").get_weights()

print(f"Updated w1: {w1}\nb1: {b1}\nw2: {w2}\nb2: {b2}")


# Assign pre-assigned weights from previous training
# considering previous weights allows the model to fine tune

w1 = np.array([[-8.94,  0.29, 12.89],[-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
w2 = np.array([[-31.38],[-27.86],[-32.79]])
b2 = np.array([15.54])

# Try a new prdiction, one data for positive and one for 
x_new = np.array([[200,16], [200,17]])
prediction_new = model.predict(x_new)
print(f"Prediction -> {prediction_new}")

# Convert prediction to decision 1 or 0
decision = np.zeros_like(prediction_new)
for i in range(len(prediction_new)):
    if prediction_new[i] >= 0.5:
        decision[i] = 1
    else:
        decision[i] = 0
print(f"decisions = {decision}")

