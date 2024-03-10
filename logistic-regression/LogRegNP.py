import numpy as np
import copy, math

# ================================
# Required Utilities
# ================================

# regression function to calculate z
def regression(x,w,b):
  return np.dot(x,w) + b

# sigmoid function
def sigmoid(z):
  return 1/(1+np.exp(-z))


# cost function
def log_cost_function(x,y,w,b):
  m = x.shape[0]
  cost = 0.0
  for i in range(m):
    z_i = regression(x[i],w,b)
    g_z_i = sigmoid(z_i)
    cost += -y[i]*np.log(g_z_i) - (1-y[i])*np.log(1-g_z_i)
  
  cost = cost / m
  return cost

# Graident Derivative calculation
# First define function to find the partial derivates for dw and db
def log_gradient_derivates(x,y,w,b):
  m,n = x.shape
  dj_dw = np.zeros(n,)
  dj_db = 0.

  for i in range(m):
    f_wb_i = sigmoid(regression(x[i],w, b))
    err_i = f_wb_i - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err_i * x[i,j]
    dj_db = dj_db + err_i
  
  dj_db = dj_db / m
  dj_dw = dj_dw / m

  return dj_dw, dj_db

# Gradient Descent
# Second define the gradient descent functions using partial derivates from above function
def log_gradient_descent (x,y,w,b,alpha,num_iters):
  j_history = []
  w_tmp = copy.deepcopy(w) # do not modify global w
  
  for i in range(num_iters):
    dj_dw_val, dj_db_val = log_gradient_derivates(x,y,w_tmp,b)
    w_tmp = w_tmp - (alpha * dj_dw_val) 
    b = b - (alpha * dj_db_val)

    # intermittent save to prevent resource exhaustion
    if i < 10000:
      j_history.append(log_cost_function(x,y,w_tmp,b))
    
    if i% math.ceil(num_iters / 10) == 0:
      print(f"Iteration {i}: cost -> {j_history[-1]}")
  
  return w_tmp, b, j_history

# ==================================

# Example data
x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3,0.5], [2,2], [1,2.5]])
y_train = np.array([0,0,0,1,1,1])

# Test Run Gradient Descent derivates on the example values
w_in = np.array([2.,3.])
b_in = 1.
dj_dw_val, dj_db_val = log_gradient_derivates(x_train, y_train, w_in, b_in)
print(f"dj_dw_val = {dj_dw_val}\ndj_db_val = {dj_db_val}")


# Test run gradient descent
w = np.zeros_like(x_train[0])
b = 0.
alpha = 0.1
iter = 10000

w_out, b_out, history = log_gradient_descent(x_train, y_train, w, b, alpha, iter)
print(f"updated paramters: w = {w_out}, b = {b_out}")

# Prediction of a new value based on updated parameters
x_new = np.array([[2.1, 0.7]]) 
print(f'Prediction of a new value based on updated parameters -> \n x: {x_new}\ny: {sigmoid(regression(x_new,w_out,b_out))}')



