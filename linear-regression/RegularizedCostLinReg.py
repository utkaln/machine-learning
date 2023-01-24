import numpy as np
np.set_printoptions(precision=8)

def regulated_cost_calc(w,m,n,l):
    regulated_cost = 0
    for j in range(n):
        regulated_cost += (w[j]**2)
    regulated_cost = regulated_cost * (l / (2*m))
    return regulated_cost

def compute_cost_lin_reg(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost += (f_wb_i - y[i])**2
    cost = cost / (2*m)
    return cost

def regularized_lin_reg_cost(x,y,w,b,l):
    m = x.shape[0]
    n = len(w)
    cost = compute_cost_lin_reg(x,y,w,b)
    regulated_cost = regulated_cost_calc(w,m,n,l)
    
    total_cost = cost + regulated_cost
    return total_cost




# Run a test to view the cost value for example data
np.random.seed(1)
x = np.random.rand(5,3)
y = np.array([0,1,0,1,0])
w = np.random.rand(x.shape[1]).reshape(-1,)-0.5
b = 0.5
l = 0.7
print(f" input vals :\nx: {x}\ny: {y}\nw: {w}\nb: {b}\nl: {l}")

cost_calc = regularized_lin_reg_cost(x,y,w,b,l)
print(f"cost found to be -> {cost_calc}")


# Gradient Descent Calculation
def calc_derivative_regularized_lin(x,y,w,b,l):
    dj_dw, dj_db = derivative_lin_reg(x,y,w,b)
    m,n = x.shape
    dj_dw_reg = regularized_derivates(dj_dw,l,m,n)
    return dj_dw_reg, dj_db

def derivative_lin_reg(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i],w) + b) - y[i]
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
dj_dw_reg, dj_db  = calc_derivative_regularized_lin(x,y,w,b,l)
print(f"dj_db: {dj_db}\ndj_dw_regularized: {dj_dw_reg.tolist()}")