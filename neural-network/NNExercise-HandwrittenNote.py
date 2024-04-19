# This exercise is to recognize two handwritten digits, zero or one
# It is done using tensorflow classification model
# The model has Three dense layers
# First layer: 25 neurons, Second layer: 15 neurons, Third layer: 1 neuron
# Dimension of W : Neuron_in X Neuron_out
# Dimension of B: Neuron_out
# W1: 400 x 25, B1: 25
# W2: 25 x 15, B2: 15
# W3: 15 x 1, B3: 1

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential, losses, optimizers

model = Sequential(
    [
        Input(shape=(400,)),
        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1, activation='sigmoid', name='layer3'),
    ]
)

model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(0.001),
)

# model.fit
# (X,y,
#  epochs=20)


