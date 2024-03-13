import numpy as np
from LogRegCostFunction import log_reg_cost_fn, sigmoid_fn

def regulated_cost_calc(w,m,n,l):
    regulated_cost = 0
    for j in range(n):
        regulated_cost += (w[j]**2)
    regulated_cost = regulated_cost * (l / (2*m))
    return regulated_cost


def regularized_lin_reg_cost(x,y,w,b,l):
    m = x.shape[0]
    n = len(w)
    cost = log_reg_cost_fn(x,y,w,b)
    regulated_cost = regulated_cost_calc(w,m,n,l)
    
    total_cost = cost + regulated_cost
    return total_cost



# Run a test to view the cost value for example data
np.random.seed(1)
x = np.random.rand(5,6)
y = np.array([0,1,0,1,0])
w = np.random.rand(x.shape[1]).reshape(-1,)-0.5
b = 0.5
l = 0.7
print(f" input vals :\nx: {x}\ny: {y}\nw: {w}\nb: {b}\nl: {l}")

cost_calc = regularized_lin_reg_cost(x,y,w,b,l)
print(f"cost found to be -> {cost_calc}")



# Gradient Descent Calculation
def calc_derivative_regularized_log(x,y,w,b,l):
    dj_dw, dj_db = derivative_log_reg(x,y,w,b)
    m,n = x.shape
    dj_dw_reg = regularized_derivates(dj_dw,l,m,n)
    return dj_dw_reg, dj_db

def derivative_log_reg(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    for i in range(m):
        z_i = np.dot(x[i],w) + b
        f_wb_i = sigmoid_fn(z_i)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def regularized_derivates(dj_dw,l,m,n):
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (l/m) * w[j]
    return dj_dw


# Try with sample values
dj_dw_reg, dj_db  = calc_derivative_regularized_log(x,y,w,b,l)
print(f"dj_db: {dj_db}\ndj_dw_regularized: {dj_dw_reg.tolist()}")


# Prediction
def predict(x, w, b): 
    # number of training examples
    m, n = x.shape   
    p = np.zeros(m)
   
    # Loop over each example
    for i in range(m):   
        z_wb = np.dot(x[i],w) + b
        f_wb = sigmoid_fn(z_wb)
        

        # Apply the threshold
        p[i] = f_wb >= 0.5
        
    return p

