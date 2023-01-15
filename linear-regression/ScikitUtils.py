import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from MultiLinearRegression import loadHouseData

x_train, y_train = loadHouseData()
x_features = ['size', 'bedroom', 'floors', 'age']

# normalize training data
scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)

print(f"before normalization data -> {x_train}")
print(f"after normalization data -> {x_norm}")

# run linear regression
sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_norm, y_train)
w_val = sgdr.coef_
b_val = sgdr.intercept_
print(f"sgdr iterations -> {sgdr.n_iter_} ")
print(f"b -> {b_val} \nw -> {w_val}")

# make prediction using sgdr
y_sgdr = sgdr.predict(x_norm)

# make prediction using w,b values
y_regression = np.dot(x_norm, w_val) + b_val

print(f"y_sgdr -> {y_sgdr} \ny_regression -> {y_regression} \ny_train -> {y_train}")
