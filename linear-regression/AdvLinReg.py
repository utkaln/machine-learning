import numpy as np
import tensorflow as tf
import ScikitUtils as skutil


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# +++++++++++++++++++++++++++++++++++++++++++++++
# UTILITIES
# +++++++++++++++++++++++++++++++++++++++++++++++


np.set_printoptions(precision=2)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

def load_sample_data():
  return np.array([[1651., 432.65], [1691.82, 454.94], [1732.63, 471.53], [1773.45, 482.51],[1814.27, 468.36], [1855.08, 482.15], [1895.9, 540.02], [1936.71, 534.58], [1977.53, 558.35], [2018.35, 566.42], [2059.16, 581.4 ], [2099.98, 596.46], [2140.8, 596.71], [2181.61, 619.45], [2222.43, 616.58], [2263.24, 653.16], [2304.06, 666.52], [2344.88, 670.59], [2385.69, 669.02], [2426.51, 678.91], [2467.33, 707.44], [2508.14, 710.76], [2548.96, 745.19], [2589.78, 729.85], [2630.59, 743.8 ], [2671.41, 738.2 ], [2712.22, 772.95], [2753.04, 772.22], [2793.86, 784.21], [2834.67, 776.43], [2875.49, 804.78], [2916.31, 833.27], [2957.12, 825.69], [2997.94, 821.05], [3038.76, 833.82], [3079.57, 833.06], [3120.39, 825.7 ], [3161.2, 843.58], [3202.02, 869.4 ], [3242.84, 851.5], [3283.65, 863.18], [3324.47, 853.01], [3365.29, 877.16], [3406.1, 863.74], [3446.92,  874.67], [3487.73, 877.74], [3528.55, 874.11], [3569.37, 882.8 ], [3610.18, 910.83], [3651., 897.42]])

# Split the data to x and y and make it Two dimensional
def transform_data(data):
  x = data[:,0]
  y = data[:,1]
  x = np.expand_dims(x, axis=1)
  y = np.expand_dims(y, axis=1)
  return x,y


# mean squared error calculation
# this is an alternative to using sci kit function
def mse_manual(y_hat, y_train):
  total_sq_err = 0
  for i in range(len(y_hat)):
    sq_err_i = (y_hat[i] - y_train)**2
    total_sq_err += sq_err_i
  return total_sq_err / (2*len(y_hat))


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 1 - Create Dataset
# +++++++++++++++++++++++++++++++++++++++++++++++

# Load sample data in a matrix structure
x_data, y_data = transform_data(load_sample_data())


# Split data to training set, validation set and test set

# Split 60% of data to training and remainder to temporary
x_train, x_, y_train, y_ = train_test_split(x_data, y_data, test_size=0.40, random_state=1)

# Split the remaining 40% of data from above into validation and test
x_val, x_test, y_val, y_test = train_test_split(x_,y_,test_size=0.5, random_state=1)

del x_, y_

print(f"Shape of X -> {x_data.shape}\nShape of Y -> {y_data.shape}")
print(f"Shape of x_train -> {x_train.shape}\nShape of y_train -> {y_train.shape}")
print(f"Shape of x_val -> {x_val.shape}\nShape of y_val -> {y_val.shape}")
print(f"Shape of x_test -> {x_test.shape}\nShape of y_test -> {y_test.shape}")


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 2 - Develop Model for Linear Regression
# +++++++++++++++++++++++++++++++++++++++++++++++

# Feature Scaling by computing z score
# z = (x - mean)/ (std_dev)

# Initialize class
scaler_linear = StandardScaler()

# Compute mean and std dev of training data and transform
x_train_scaled = scaler_linear.fit_transform(x_train)
print("Training Data Tranformation -->")
print(f"mean -> {scaler_linear.mean_}")
print(f"std dev -> {scaler_linear.scale_}")

# Train model
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)

# Initial prediction
y_hat = linear_model.predict(x_train_scaled)
mse = mean_squared_error(y_true=y_train, y_pred=y_hat)/2 # scikit returns without dividing by 2, hence done
print(f"mse on training -> {mse}")

# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 3 - Evaluate Model 
# +++++++++++++++++++++++++++++++++++++++++++++++


# Feature scaling of validation data
# Remember to feature scale using training data, so the same scale applies on validation data
# DO NOT feature scale using validation data, as it would create inaccurate scaling compared to training data
# To do this use tranform() method of sci kit instead of fit_transform() from before
x_val_scaled = scaler_linear.transform(x_val)

print("Validation Data Tranformation -->")
print(f"mean -> {scaler_linear.mean_}")
print(f"std dev -> {scaler_linear.scale_}")

# Enhanced y_hat with validation data
y_hat = linear_model.predict(x_val_scaled)

# Measure prediction error between training data and validation data using Mean Squared Error (MSE)
# j_wb = (1/(2*count))* sum of ((x_train[i] - y_train[i])**2)
mse = mean_squared_error(y_true=y_val, y_pred=y_hat)/2
print(f"mse on validation -> {mse}")


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 4 - Add Polynomial Features 
# +++++++++++++++++++++++++++++++++++++++++++++++

# Generate polynomial features from training set with degree =2
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly_transformed = poly.fit_transform(x_train)

# Scale inputs
scaler_poly = StandardScaler()
x_train_poly_scaled = scaler_poly.fit_transform(x_train_poly_transformed)

# Train the model
model = LinearRegression()
model.fit(x_train_poly_scaled, y_train)
y_hat = model.predict(x_train_poly_scaled)
mse = mean_squared_error(y_true=y_train, y_pred=y_hat)/2
print(f"mse on poly training -> {mse}")

# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 5 - Evaluate Model with Polynomial Features 
# +++++++++++++++++++++++++++++++++++++++++++++++
poly = PolynomialFeatures(degree=2, include_bias=False)
x_val_poly = poly.fit_transform(x_val)

# Scale Features
x_val_poly_scaled = scaler_poly.transform(x_val_poly)

# Predict with poly data scaled
y_hat = model.predict(x_val_poly_scaled)
mse = mean_squared_error(y_true=y_val, y_pred=y_hat)/2
print(f"mse on poly validation -> {mse}")


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 6 - Try out different Polynomial degrees  
# +++++++++++++++++++++++++++++++++++++++++++++++

models = []
scalers = []
train_mses = []
val_mses = []

# Try polynomials up to degree 10
for degree in range(1,11):
  poly = PolynomialFeatures(degree=degree, include_bias=False)
  x_train_poly_transformed = poly.fit_transform(x_train)

  # Scale inputs
  scaler_poly = StandardScaler()
  x_train_poly_scaled = scaler_poly.fit_transform(x_train_poly_transformed)
  scalers.append(scaler_poly)

  # Train the model
  model = LinearRegression()
  model.fit(x_train_poly_scaled, y_train)
  models.append(model)

  # Prediction and MSE
  y_hat = model.predict(x_train_poly_scaled)
  train_mse = mean_squared_error(y_true=y_train, y_pred=y_hat)/2
  train_mses.append(train_mse)

  # Add polynomial features and scale the cross validation set
  poly = PolynomialFeatures(degree, include_bias=False)
  x_val_transformed = poly.fit_transform(x_val)
  x_val_scaled = scaler_poly.transform(x_val_transformed)

  # Prediction and MSE with Validation data
  y_hat = model.predict(x_val_scaled)
  val_mse = mean_squared_error(y_true=y_val, y_pred=y_hat)/2
  val_mses.append(val_mse)


# +++++++++++++++++++++++++++++++++++++++++++++++
# Step 7 - Choose the best model
# +++++++++++++++++++++++++++++++++++++++++++++++

# choose the model with lowest validation MSE
lowest_poly = np.argmin(val_mses) + 1
print(f"Lowest Error with Degrees: {lowest_poly}")





