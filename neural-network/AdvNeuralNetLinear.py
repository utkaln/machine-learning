import numpy as np
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from BuildNNModels import build_nn_models


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

# +++++++++++++++++++++++++++++++++++++++++++++++
# Try polynomial feature with Degree = 1
# +++++++++++++++++++++++++++++++++++++++++++++++

# With degree 1 it is like linear regression
degree = 1
poly = PolynomialFeatures(degree=degree, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
x_val_mapped = poly.transform(x_val)
x_test_mapped = poly.transform(x_test)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_mapped)
x_val_scaled = scaler.transform(x_val_mapped)
x_test_scaled = scaler.transform(x_test_mapped)

# Build and Train models
nn_train_mses = []
nn_val_mses = []

nn_models = build_nn_models()

for model in nn_models:
  # Setup loss function and optimizer 
  model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),)
  
  # Train the model
  model.fit(x_train_scaled, y_train, epochs=300, verbose=0 )

  # Prediction on Training data
  y_hat = model.predict(x_train_scaled)
  train_mse = mean_squared_error(y_true=y_train, y_pred=y_hat)/2
  print(f"Training MSE found as : {train_mse} , for {model.name}")
  nn_train_mses.append(train_mse)

  # Prediction on Validation data
  y_hat = model.predict(x_val_scaled)
  val_mse = mean_squared_error(y_true=y_val, y_pred=y_hat)/2
  print(f"Validation  MSE found as : {val_mse} , for {model.name}")
  nn_val_mses.append(val_mse)


lowest_mse = np.argmin(nn_val_mses) 
lowest_model = nn_models[lowest_mse]
print(f"Model with Lowest MSE = {lowest_model.name}")

# Prediction from the model with Lowest MSE
y_hat = lowest_model.predict(x_test_scaled)
test_mse = mean_squared_error(y_true=y_test, y_pred=y_hat)

print(f"Test MSE for Model - {lowest_model.name} is: {test_mse}")


