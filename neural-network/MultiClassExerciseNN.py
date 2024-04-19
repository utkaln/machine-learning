import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential, losses, optimizers

# NumPy Implementation of Softmax
# receives z values where z = wx + b
# returns a value for next layer a = e**z(i) / sum(e**z(n))
def softmax_fn(z):
    # Initialize a as empty array
    a = np.empty(0)
    
    # Initialize denominator value to 0
    denominator = 0
    for i in range(len(z)):
        denominator += np.exp(z[i])
    
    # Create output array by iterating over z values and dividing by denominator
    for j in range(len(z)):
        a = np.append(a,np.exp(z[j])/denominator)
    
    return a

# Test softmax numpy implementation
z = np.array([1.,2.,3.,4.])
a_np = softmax_fn(z)

# tensorflow calculation of softmax
a_tf = tf.nn.softmax(z)

print(f'a_np -> {a_np}\na_tf -> {a_tf}')

nn_model = Sequential(
    [
        Input(shape=(400,)),
        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1, activation='sigmoid', name='layer3'),
    ]
)

nn_model.summary()

nn_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizers.Adam(0.001),
)


# prediction_val = nn_model.predict(X.reshape(1,400))
# prediction_probability = tf.nn.softmax(prediction_val)

