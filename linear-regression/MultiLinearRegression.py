import copy, math
import numpy as np
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Factors 
def loadHouseData():
  x = np.array([[2104, 5, 1, 45],[1416, 3, 2, 40],[852, 2, 1, 35]])
  y = np.array([460, 232, 178])
  return x, y


x_train, y_train = loadHouseData()
print(f"x_train -> {x_train}")
print(f"y_train -> {y_train}")

# Parameters : b is a scalar and w is a vector
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# Prediction equation using numpy vector
def predictVector (x,w,b):
  return np.dot(x,w) + b

# Error Prediction function to be used in Gradient Descent calculation
def errorPrediction(x,y,w,b):
  return (predictVector (x,w,b) - y)

x_vector = x_train[0,:]
print(f"x_vector -> {x_vector}")

y_prediction = predictVector(x_vector, w_init, b_init)
print(f"y_prediction -> {y_prediction}")


# Cost Function
def costFunction(x,y,w,b):
  m = x.shape[0]
  cost = 0.0
  for i in range(m):
    predict_i = predictVector(x[i], w, b)
    cost = cost + (predict_i - y[i])**2
  cost = cost / (2 * m)
  return cost

cost = costFunction(x_train, y_train, w_init, b_init)
print(f"cost -> {cost}")


# Derivative required in Gradient Descent Calculation for Multi variables
def computeDerivative(x,y,w,b):
  m,n = x.shape

  # initialize partial derivative w and partial derivative b
  dj_dw = np.zeros((n,))
  dj_db = 0.

  for i in range(m):
    error_pred = errorPrediction(x[i], y[i], w, b)
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
      cost_history.append(costFunction(x,y,w_temp,b))
    
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
  print(f"new prediction -> {predictVector(x_train[i], w_final, b_final)} \n Target Value -> {y_train[i]}")





