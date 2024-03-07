import copy, math
import numpy as np
from utilities.LoadData import loadHouseData, preInitWb
from utilities.RegressionUtils import predict_regression, cost_regression, error_regression
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


x_train, y_train = loadHouseData()

# Parameters : b is a scalar and w is a vector
b_init, w_init = preInitWb()

x_vector = x_train[0,:]
print(f"x_vector -> {x_vector}")

y_prediction = predict_regression(x_vector, w_init, b_init)
print(f"y_prediction -> {y_prediction}")


cost = cost_regression(x_train, y_train, w_init, b_init)
print(f"cost -> {cost}")


# Derivative required in Gradient Descent Calculation for Multi variables
def computeDerivative(x,y,w,b):
  m,n = x.shape

  # initialize partial derivative w and partial derivative b
  dj_dw = np.zeros((n,))
  dj_db = 0.

  for i in range(m):
    error_pred = error_regression(x[i], y[i], w, b)
    dj_db = dj_db + error_pred
    for j in range(n):
      dj_dw[j] = dj_dw[j] + error_pred * x[i,j]
  dj_db = dj_db / m
  dj_dw = dj_dw / m

  return dj_db, dj_dw


# Print the result of Gradient Descent
tmp_dj_db, tmp_dj_dw = computeDerivative(x_train,y_train,w_init,b_init)
print(f"tmp_dj_db -> {tmp_dj_db} ")
print(f"tmp_dj_dw -> {tmp_dj_dw} ")


# Gradient Descent Calculation
def gradientDescent(x,y,w,b, alpha, iterations):
  cost_history = []
  w_temp = copy.deepcopy(w) # avoid modifying global w 

  for i in range(iterations):
    dj_db, dj_dw = computeDerivative(x,y,w_temp,b)
    w_temp = w_temp - (alpha * dj_dw)
    b = b - (alpha * dj_db)

    # Intermediate append to save from resource exhaustion
    if i<10000:
      cost_history.append(cost_regression(x,y,w_temp,b))
    
    print(f"Iteration -> {i}, cost -> {cost_history}")
  
  return w_temp, b, cost_history


# initialize variables and invoke gradient descent call
initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 10
alpha = 5.0e-7

w_final, b_final, cost_history = gradientDescent(x_train, y_train, initial_w, initial_b, alpha, iterations)
print(f"w_final -> {w_final}, \nb_final -> {b_final}")
m,_ = x_train.shape
for i in range(m):
  print(f"new prediction -> {predict_regression(x_train[i], w_final, b_final)} \n Target Value -> {y_train[i]}")





