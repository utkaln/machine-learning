import numpy as np
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

# +++++++++++++++++++++++++++++++++++++++++++++++
# UTILITIES
# +++++++++++++++++++++++++++++++++++++++++++++++


def load_classification_data():
  return np.array([[1.00e+03,1.46e+03,0.00e+00]
,[1.05e+03,1.01e+03,0.00e+00]
,[1.09e+03,8.54e+02,0.00e+00]
,[1.14e+03,2.56e+03,0.00e+00]
,[1.18e+03,4.62e+03,0.00e+00]
,[1.23e+03,4.97e+03,0.00e+00]
,[1.27e+03,2.39e+03,0.00e+00]
,[1.32e+03,1.01e+02,0.00e+00]
,[1.36e+03,7.29e+02,0.00e+00]
,[1.41e+03,4.22e+03,0.00e+00]
,[1.45e+03,4.30e+03,0.00e+00]
,[1.50e+03,4.52e+02,0.00e+00]
,[1.54e+03,2.76e+02,0.00e+00]
,[1.59e+03,2.24e+03,0.00e+00]
,[1.63e+03,2.76e+03,0.00e+00]
,[1.68e+03,2.96e+03,0.00e+00]
,[1.72e+03,3.99e+03,0.00e+00]
,[1.77e+03,8.79e+02,0.00e+00]
,[1.81e+03,3.42e+03,0.00e+00]
,[1.86e+03,1.48e+03,0.00e+00]
,[1.90e+03,1.28e+03,0.00e+00]
,[1.95e+03,4.02e+02,0.00e+00]
,[1.99e+03,1.11e+03,0.00e+00]
,[2.04e+03,2.36e+03,0.00e+00]
,[2.09e+03,7.79e+02,1.00e+00]
,[2.13e+03,4.07e+03,0.00e+00]
,[2.18e+03,9.55e+02,0.00e+00]
,[2.22e+03,7.04e+02,1.00e+00]
,[2.27e+03,4.85e+03,0.00e+00]
,[2.31e+03,6.78e+02,0.00e+00]
,[2.36e+03,1.18e+03,1.00e+00]
,[2.40e+03,4.15e+03,0.00e+00]
,[2.45e+03,4.87e+03,0.00e+00]
,[2.49e+03,4.45e+03,0.00e+00]
,[2.54e+03,4.42e+03,0.00e+00]
,[2.58e+03,2.44e+03,0.00e+00]
,[2.63e+03,4.37e+03,0.00e+00]
,[2.67e+03,1.83e+03,1.00e+00]
,[2.72e+03,1.73e+03,0.00e+00]
,[2.76e+03,4.32e+03,0.00e+00]
,[2.81e+03,2.71e+03,0.00e+00]
,[2.85e+03,2.69e+03,0.00e+00]
,[2.90e+03,4.75e+03,0.00e+00]
,[2.94e+03,3.52e+02,1.00e+00]
,[2.99e+03,1.41e+03,0.00e+00]
,[3.04e+03,4.77e+02,0.00e+00]
,[3.08e+03,2.86e+03,0.00e+00]
,[3.13e+03,9.80e+02,0.00e+00]
,[3.17e+03,4.65e+03,0.00e+00]
,[3.22e+03,3.12e+03,0.00e+00]
,[3.26e+03,2.46e+03,0.00e+00]
,[3.31e+03,3.09e+03,0.00e+00]
,[3.35e+03,2.99e+03,0.00e+00]
,[3.40e+03,1.33e+03,1.00e+00]
,[3.44e+03,8.29e+02,1.00e+00]
,[3.49e+03,4.50e+03,0.00e+00]
,[3.53e+03,4.55e+03,0.00e+00]
,[3.58e+03,2.66e+03,1.00e+00]
,[3.62e+03,5.00e+03,0.00e+00]
,[3.67e+03,3.47e+03,0.00e+00]
,[3.71e+03,2.91e+03,0.00e+00]
,[3.76e+03,1.68e+03,0.00e+00]
,[3.80e+03,1.96e+03,0.00e+00]
,[3.85e+03,1.06e+03,0.00e+00]
,[3.89e+03,4.27e+02,0.00e+00]
,[3.94e+03,1.26e+02,1.00e+00]
,[3.98e+03,3.19e+03,0.00e+00]
,[4.03e+03,2.64e+03,1.00e+00]
,[4.08e+03,1.21e+03,1.00e+00]
,[4.12e+03,1.66e+03,1.00e+00]
,[4.17e+03,1.36e+03,0.00e+00]
,[4.21e+03,2.11e+03,0.00e+00]
,[4.26e+03,4.60e+03,0.00e+00]
,[4.30e+03,3.97e+03,0.00e+00]
,[4.35e+03,4.17e+03,0.00e+00]
,[4.39e+03,2.84e+03,1.00e+00]
,[4.44e+03,3.02e+02,1.00e+00]
,[4.48e+03,2.94e+03,0.00e+00]
,[4.53e+03,2.34e+03,1.00e+00]
,[4.57e+03,3.02e+03,0.00e+00]
,[4.62e+03,3.87e+03,0.00e+00]
,[4.66e+03,2.26e+03,0.00e+00]
,[4.71e+03,2.04e+03,0.00e+00]
,[4.75e+03,3.07e+03,1.00e+00]
,[4.80e+03,4.80e+03,0.00e+00]
,[4.84e+03,3.27e+02,1.00e+00]
,[4.89e+03,2.06e+03,0.00e+00]
,[4.93e+03,3.32e+03,1.00e+00]
,[4.98e+03,4.70e+03,0.00e+00]
,[5.03e+03,1.13e+03,0.00e+00]
,[5.07e+03,2.49e+03,1.00e+00]
,[5.12e+03,9.05e+02,1.00e+00]
,[5.16e+03,4.05e+03,0.00e+00]
,[5.21e+03,4.67e+03,0.00e+00]
,[5.25e+03,3.84e+03,0.00e+00]
,[5.30e+03,2.59e+03,1.00e+00]
,[5.34e+03,4.90e+03,0.00e+00]
,[5.39e+03,4.95e+03,0.00e+00]
,[5.43e+03,3.72e+03,1.00e+00]
,[5.48e+03,4.35e+03,0.00e+00]
,[5.52e+03,1.88e+03,1.00e+00]
,[5.57e+03,5.28e+02,1.00e+00]
,[5.61e+03,2.29e+03,1.00e+00]
,[5.66e+03,3.82e+03,1.00e+00]
,[5.70e+03,5.03e+01,1.00e+00]
,[5.75e+03,1.76e+03,1.00e+00]
,[5.79e+03,2.14e+03,1.00e+00]
,[5.84e+03,3.77e+03,1.00e+00]
,[5.88e+03,1.51e+02,1.00e+00]
,[5.93e+03,2.81e+03,0.00e+00]
,[5.97e+03,0.00e+00,1.00e+00]
,[6.02e+03,3.89e+03,0.00e+00]
,[6.07e+03,1.93e+03,1.00e+00]
,[6.11e+03,1.63e+03,1.00e+00]
,[6.16e+03,1.38e+03,1.00e+00]
,[6.20e+03,4.20e+03,1.00e+00]
,[6.25e+03,2.21e+03,1.00e+00]
,[6.29e+03,3.27e+03,1.00e+00]
,[6.34e+03,1.16e+03,1.00e+00]
,[6.38e+03,1.56e+03,1.00e+00]
,[6.43e+03,1.86e+03,1.00e+00]
,[6.47e+03,2.31e+03,1.00e+00]
,[6.52e+03,3.69e+03,0.00e+00]
,[6.56e+03,4.02e+03,1.00e+00]
,[6.61e+03,3.59e+03,1.00e+00]
,[6.65e+03,2.19e+03,1.00e+00]
,[6.70e+03,4.52e+03,0.00e+00]
,[6.74e+03,3.64e+03,0.00e+00]
,[6.79e+03,4.12e+03,0.00e+00]
,[6.83e+03,2.51e+02,1.00e+00]
,[6.88e+03,8.04e+02,1.00e+00]
,[6.92e+03,2.09e+03,1.00e+00]
,[6.97e+03,4.57e+03,1.00e+00]
,[7.02e+03,2.51e+03,1.00e+00]
,[7.06e+03,3.14e+03,1.00e+00]
,[7.11e+03,5.78e+02,1.00e+00]
,[7.15e+03,3.17e+03,1.00e+00]
,[7.20e+03,2.26e+02,1.00e+00]
,[7.24e+03,4.27e+03,1.00e+00]
,[7.29e+03,2.61e+03,1.00e+00]
,[7.33e+03,3.79e+03,1.00e+00]
,[7.38e+03,3.39e+03,1.00e+00]
,[7.42e+03,2.79e+03,1.00e+00]
,[7.47e+03,4.72e+03,1.00e+00]
,[7.51e+03,1.61e+03,1.00e+00]
,[7.56e+03,3.77e+02,1.00e+00]
,[7.60e+03,1.03e+03,1.00e+00]
,[7.65e+03,4.10e+03,1.00e+00]
,[7.69e+03,2.74e+03,1.00e+00]
,[7.74e+03,2.01e+03,1.00e+00]
,[7.78e+03,1.31e+03,1.00e+00]
,[7.83e+03,6.53e+02,1.00e+00]
,[7.87e+03,1.91e+03,1.00e+00]
,[7.92e+03,1.08e+03,1.00e+00]
,[7.96e+03,6.03e+02,1.00e+00]
,[8.01e+03,7.54e+01,1.00e+00]
,[8.06e+03,4.25e+03,1.00e+00]
,[8.10e+03,1.23e+03,1.00e+00]
,[8.15e+03,3.74e+03,1.00e+00]
,[8.19e+03,3.29e+03,1.00e+00]
,[8.24e+03,4.77e+03,1.00e+00]
,[8.28e+03,7.54e+02,1.00e+00]
,[8.33e+03,3.04e+03,1.00e+00]
,[8.37e+03,2.89e+03,1.00e+00]
,[8.42e+03,4.40e+03,1.00e+00]
,[8.46e+03,2.01e+02,1.00e+00]
,[8.51e+03,1.51e+03,1.00e+00]
,[8.55e+03,3.22e+03,1.00e+00]
,[8.60e+03,2.51e+01,1.00e+00]
,[8.64e+03,1.43e+03,1.00e+00]
,[8.69e+03,5.53e+02,1.00e+00]
,[8.73e+03,1.53e+03,1.00e+00]
,[8.78e+03,1.58e+03,1.00e+00]
,[8.82e+03,1.76e+02,1.00e+00]
,[8.87e+03,4.92e+03,1.00e+00]
,[8.91e+03,3.54e+03,1.00e+00]
,[8.96e+03,2.16e+03,1.00e+00]
,[9.01e+03,2.41e+03,1.00e+00]
,[9.05e+03,1.71e+03,1.00e+00]
,[9.10e+03,1.26e+03,1.00e+00]
,[9.14e+03,3.57e+03,1.00e+00]
,[9.19e+03,3.94e+03,1.00e+00]
,[9.23e+03,3.92e+03,1.00e+00]
,[9.28e+03,3.49e+03,1.00e+00]
,[9.32e+03,3.67e+03,1.00e+00]
,[9.37e+03,2.54e+03,1.00e+00]
,[9.41e+03,5.03e+02,1.00e+00]
,[9.46e+03,4.47e+03,1.00e+00]
,[9.50e+03,6.28e+02,1.00e+00]
,[9.55e+03,3.37e+03,1.00e+00]
,[9.59e+03,1.78e+03,1.00e+00]
,[9.64e+03,3.24e+03,1.00e+00]
,[9.68e+03,3.62e+03,1.00e+00]
,[9.73e+03,4.82e+03,1.00e+00]
,[9.77e+03,1.98e+03,1.00e+00]
,[9.82e+03,3.34e+03,1.00e+00]
,[9.86e+03,3.44e+03,1.00e+00]
,[9.91e+03,1.81e+03,1.00e+00]
,[9.95e+03,3.52e+03,1.00e+00]
,[1.00e+04,9.30e+02,1.00e+00]])

def format_data(data):
  x = data[:,:-1]
  y = data[:,-1]
  y = np.expand_dims(y, axis=1)
  return x,y



# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 1: Load Data and Split
# +++++++++++++++++++++++++++++++++++++++++++++++


# Load data
x,y = format_data(load_classification_data())

# Split dataset to Training, Validation, Test
x_train, x_, y_train, y_ = train_test_split(x,y,test_size=0.4, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_,y_,test_size=0.5, random_state=1)
del x_, y_


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 2: Build and Train Model with regularization
# +++++++++++++++++++++++++++++++++++++++++++++++

nn_train_err = []
nn_val_err = []

lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models = [None] * len(lambdas)

for i in range(len(lambdas)):

  lambda_ = lambdas[i]
  
  models[i] = Sequential(
    [
      # Incorrect syntax - not working
      # Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.12(lambda_)),
      # Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.12(lambda_)),
      # Dense(classes, activation='linear'),
    ]   
  )

  models[i].compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  )
  
  models[i].fit(x_train, y_train, epochs=200, verbose=0)

  # Set the threshold for classification 0.5
  threshold = 0.5

  # Compute fraction of misclassified data from training set
  y_hat = models[i].predict(x_train)
  y_hat = tf.math.sigmoid(y_hat)
  y_hat = np.where(y_hat >= threshold, 1, 0)
  train_err = np.mean(y_hat != y_train)
  print(f"train err -> {train_err}")
  nn_train_err.append(train_err)

  # Compute fraction of misclassified data from validation set
  y_hat = models[i].predict(x_val)
  y_hat = tf.math.sigmoid(y_hat)
  y_hat = np.where(y_hat >= threshold, 1, 0)
  val_err = np.mean(y_hat != y_val)
  print(f"val err --> {val_err}")
  nn_val_err.append(val_err)


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 3: Predict on Test data from the lowest err
# +++++++++++++++++++++++++++++++++++++++++++++++

lowest_err = np.argmin(nn_val_err) 
lowest_model = models[lowest_err]
print(f"Model with Lowest Error = {lowest_model.name}")

# Prediction from the model with Lowest MSE
y_hat = lowest_model.predict(x_test)
y_hat = tf.math.sigmoid(y_hat)
y_hat = np.where(y_hat >= threshold, 1, 0)
test_err = np.mean(y_hat != y_test)
print(f"Lowest Test Err --> {test_err}")




