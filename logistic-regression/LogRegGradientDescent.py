import copy, math
import numpy as np
from LogRegCostFunction import z_fn, sigmoid_fn, log_reg_cost_fn

# First part of Gradient Descent calculation is to find derivative
def log_reg_gradient_algorithm(X,y,w,b):
    m,n = X.shape
    # Initialize w and b
    dj_dw = np.zeros((n,)) 
    dj_db = 0.
    for i in range(m):
        sigmoid_i = sigmoid_fn(z_fn(w,X[i],b))
        err_i = sigmoid_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

# Calculate Graident Descent as batch
def log_reg_gradient_descent(X,y,w,b,alpha,num_iters):
    w_copy = copy.deepcopy(w)
    for i in range(num_iters):
        # Calculate gradient 
        dw, db = log_reg_gradient_algorithm(X,y,w_copy,b)

        # Apply learning rate to w and b
        w_copy = w_copy - alpha*dw
        b = b - alpha*db
    return w_copy, b

# Example dataset
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.zeros_like(X_tmp[0])
b_tmp = 0.
alpha = 0.1
iters = 10000
dj_dw_tmp, dj_db_tmp = log_reg_gradient_algorithm(X_tmp, y_tmp, w_tmp, b_tmp)
print("before learning ----------->")
print(f"b: {dj_db_tmp}" )
print(f"w: {dj_dw_tmp.tolist()}" )
print("after learning ----------->")
w,b = log_reg_gradient_descent(X_tmp, y_tmp, w_tmp, b_tmp, alpha, iters)
print(f"b: {b}" )
print(f"w: {w.tolist()}" )


