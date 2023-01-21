import numpy as np
import copy, math


# Define sigmoid function
def sigmoid(z):
  return 1/(1+np.exp(-z))

# Define cost function
def log_cost_function(x,y,w,b):
  m = x.shape[0]
  cost = 0.0
  for i in range(m):
    z_i = np.dot(x[i],w) + b
    f_wb_i = sigmoid(z_i)
    cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
  
  cost = cost / m
  return cost


# Example data
x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3,0.5], [2,2], [1,2.5]])
y_train = np.array([0,0,0,1,1,1])
w_in = np.array([1,1])
b_in_minus3 = -3

# Calculate Cost Function for the data above
print(f"Cost with b_in_minus3 --> {log_cost_function(x_train, y_train, w_in, b_in_minus3)}")
b_in_minus4 = -4
print(f"Cost with b_in_minus4 --> {log_cost_function(x_train, y_train, w_in, b_in_minus4)}")
b_in_minus2 = -2
print(f"Cost with b_in_minus2 --> {log_cost_function(x_train, y_train, w_in, b_in_minus2)}")
b_in_minus5 = -5
print(f"Cost with b_in_minus5 --> {log_cost_function(x_train, y_train, w_in, b_in_minus5)}")

# Graident descent calculation
# First define function to find the partial derivates for dw and db
def log_gradient_derivates(x,y,w,b):
  m,n = x.shape
  dj_dw = np.zeros(n,)
  dj_db = 0.

  for i in range(m):
    f_wb_i = sigmoid(np.dot(x[i],w) + b)
    err_i = f_wb_i - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err_i * x[i,j]
    dj_db = dj_db + err_i
  
  dj_db = dj_db / m
  dj_dw = dj_dw / m

  return dj_dw, dj_db

# Test Run Gradient Descent derivates on the example values
w_in = np.array([2.,3.])
b_in = 1.
dj_dw_val, dj_db_val = log_gradient_derivates(x_train, y_train, w_in, b_in)
print(f"dj_dw_val = {dj_dw_val}\ndj_db_val = {dj_db_val}")

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
      print(f"Interation {i}: cost -> {j_history[-1]}")
  
  return w_tmp, b, j_history


# Test run gradient descent
w = np.zeros_like(x_train[0])
b = 0.
alpha = 0.1
iter = 10000

w_out, b_out, history = log_gradient_descent(x_train, y_train, w, b, alpha, iter)
print(f"updated paramters: w = {w_out}, b = {b_out}")




#



