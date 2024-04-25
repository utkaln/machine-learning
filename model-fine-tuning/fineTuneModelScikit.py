import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras import Sequential, optimizers, losses
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from FineTuneUtils import split_data, build_nn_models


data = np.loadtxt('./data/data_1.csv', delimiter=',')
x = data[:,0]
y = data[:,1]

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

# Split Data to Train, Validation, Test sets
[x_train, y_train, x_cv, y_cv, x_test, y_test] = split_data(x,y)


# First - Fit a Linear Model 

# Apply feature scaling type z scale using scikit StandardScaler
# z = (x - mu)/ (sigma)

scaler_linear = StandardScaler()

# Mean and Standard Deviation(sigma)
X_train_scaled = scaler_linear.fit_transform(x_train)
# print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
# print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

# Train the model using Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Evaluate the Model using scikit mean squared error function
y_hat = linear_model.predict(X_train_scaled)
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, y_hat) / 2}")

# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_scaled = scaler_linear.transform(x_cv)
y_hat = linear_model.predict(X_cv_scaled)
print(f"Validation MSE (using sklearn function): {mean_squared_error(y_cv, y_hat) / 2}")

########################################################################################################

# Try the same model with polynomial features
# Instantiate the class to make polynomial features with x and x square
poly = PolynomialFeatures(degree=2, include_bias=False)

# Compute the number of features and transform the training set
X_train_mapped = poly.fit_transform(x_train)

# Feature Scale
scaler_poly = StandardScaler()
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

# Train the model
model = LinearRegression()
model.fit(X_train_mapped_scaled, y_train )
y_hat = model.predict(X_train_mapped_scaled)
print(f"Poly training MSE (using sklearn function): {mean_squared_error(y_train, y_hat) / 2}")

# Add the polynomial features to the cross validation set
X_cv_mapped = poly.transform(x_cv)
# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

# Compute the cross validation MSE
y_hat = model.predict(X_cv_mapped_scaled)
print(f"Poly Validation MSE: {mean_squared_error(y_cv, y_hat) / 2}")


########################################################################################################
# Train using Neural Network


# Prepare the data

# For inner layers Treat it like polynomial features with degree = 1, which is similar to that of linear regression
degree = 1
poly = PolynomialFeatures(degree=1, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)

# Scale the input features for faster convergence of gradient descent
# Important - Mean and standard deviation must be computed only on the scaled training data, and use the same value on CV and TEST data
scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)

# Build and Train the model

# Lists to capture error rates from training and cv data
nn_train_mse = []
nn_cv_mse = []

# build models
nn_models = build_nn_models()

# Iterate through available models
for model in nn_models:
    
    # Set loss, learning rate
    model.compile(
        loss = 'mse',
        optimizer = optimizers.Adam(learning_rate=0.1),
    )
    print(f'model training Initialized-> {model.name}')

    # Train
    model.fit(X_train_mapped_scaled, y_train, epochs=100, verbose=0)
    print(f'model training Finished-> {model.name}')

    # Save Training MSEs
    y_hat = model.predict(X_train_mapped_scaled)
    training_mse = mean_squared_error(y_train, y_hat) / 2
    nn_train_mse.append(training_mse)

    # Save Validation MSEs
    y_hat = model.predict(X_cv_mapped_scaled)
    validation_mse = mean_squared_error(y_cv, y_hat) / 2
    nn_cv_mse.append(validation_mse)

# Print results of MSEs
for model_num in range(len(nn_train_mse)):
    print(f'Model {model_num+1}: Training MSE: {nn_train_mse[model_num]: .2f}, CV MSE: {nn_cv_mse[model_num]: .2f}')


# Test out a sample model with Test Data to see if MSE hold up
y_hat = nn_models[0].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, y_hat)
print(f'Test MSE: {test_mse:.2f}')


########################################################################################################
# Train using Logistic Classification
# Dataset will have y value as 0 or 1

data = np.loadtxt('./data/data_2.csv', delimiter=',')
# Split the inputs and outputs into separate arrays
x_bc = data[:,:-1]
y_bc = data[:,-1]

# Convert y into 2-D because the commands later will require it (x is already 2-D)
y_bc = np.expand_dims(y_bc, axis=1)

# Split the dataset
[x_bc_train, y_bc_train, x_bc_cv, y_bc_cv, x_bc_test, y_bc_test] = split_data(x_bc, y_bc)

# Build and Train Model

nn_train_error = []
nn_cv_error = []

models_bc = build_nn_models()

# Loop over each model to train and validate
for model in models_bc:
    model.compile(
        loss = losses.BinaryCrossentropy(from_logits=True),
        optimizer = optimizers.Adam(learning_rate=0.1),
    )

    print(f'model training Initialized-> {model.name}')

    # Train
    model.fit(x_bc_train, y_bc_train, epochs=200, verbose=0)
    print(f'model training Finished-> {model.name}')

    # Threshold for Classification
    threshold = 0.5

    # Record error on Training data set
    
    y_hat = model.predict(x_bc_train)
    y_hat = tf.math.sigmoid(y_hat)
    y_hat = np.where(y_hat >= threshold, 1, 0)
    train_err = np.mean(y_hat != y_bc_train)
    nn_train_error.append(train_err)

    # Record error on Validation data set
    y_hat = model.predict(x_bc_cv)
    y_hat = tf.math.sigmoid(y_hat)
    y_hat = np.where(y_hat >= threshold, 1, 0)
    cv_err = np.mean(y_hat != y_bc_cv)
    nn_cv_error.append(cv_err)


# Print results of MSEs
for model_num in range(len(nn_train_error)):
    print(f'Model {model_num+1}: Training Error: {nn_train_error[model_num]: .2f}, CV Error: {nn_cv_error[model_num]: .2f}')

# Test out a sample model with Test Data to see if Error hold up
y_hat = models_bc[0].predict(x_bc_test)
y_hat = tf.math.sigmoid(y_hat)
y_hat = np.where(y_hat >= threshold, 1, 0)
test_err = np.mean(y_hat != y_bc_test)
print(f'Test Error: {test_err:.2f}')



