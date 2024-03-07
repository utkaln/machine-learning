import numpy as np
import matplotlib.pyplot as plt
from MultiLinearRegression import gradientDescent

# ======================================================
# Gradient Descent calculation with linear format wx + b

# create target data
# create a number array from 0 - 19
x = np.arange(0, 20, 1) 

# Simulate target value as square of each feature value
y = 1 + x**2

# project the x as a column array
X = x.reshape(-1, 1)
print(f"x -> {x} \n y-> {y} \n X -> {X}")

# find gradient descent 
w_init = [0.0]
b_init = 0.0
iterations = 10
alpha = 1e-2
w_final, b_final, cost_history = gradientDescent(X, y, w_init, b_init, alpha, iterations)
print(f"w -> {w_final} \nb -> {b_final} \ncost_history -> {cost_history} " )

# ======================================================
# find gradient descent with linear format wx**2 + wx + b
X = (x**2).reshape(-1, 1)
w_final, b_final, cost_history = gradientDescent(X, y, w_init, b_init, alpha, iterations)
print(f"w -> {w_final} \nb -> {b_final} \ncost_history -> {cost_history} " )


# ======================================================
# find gradient descent with linear format wx**3 + wx**2 + wx + b
X = (np.c_[x, x**2, x**3])
w_init = [0.0, 0.0, 0.0]
w_final, b_final, cost_history = gradientDescent(X, y, w_init, b_init, alpha, iterations)
print(f"w -> {w_final} \nb -> {b_final} \ncost_history -> {cost_history} " )







