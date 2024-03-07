import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def build_nn_models():
    tf.random.set_seed(20)
    model_three = Sequential(
        [
          Dense(25, activation='relu'),
          Dense(15, activation='relu'),
          Dense(1, activation='linear'),
        ],
        name= 'model_three'
    )

    model_five = Sequential(
        [
          Dense(20, activation='relu'),
          Dense(12, activation='relu'),
          Dense(12, activation='relu'),
          Dense(20, activation='relu'),
          Dense(1, activation='linear'),
        ],
        name= 'model_five'
    )

    model_six = Sequential(
        [
          Dense(32, activation='relu'),
          Dense(16, activation='relu'),
          Dense(8, activation='relu'),
          Dense(4, activation='relu'),
          Dense(12, activation='relu'),
          Dense(1, activation='linear'),
        ],
        name= 'model_six'
    )

    return [model_three, model_five, model_six]


     