import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from utilities.LoadData import loadHouseData

x_train, y_train = loadHouseData()
x_features = ['size', 'bedroom', 'floors', 'age']

# normalize training data using z score normalization
scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)

print(f"before normalization data -> {x_train}")
print(f"after normalization data -> {x_norm}")

# run linear regression
# gradient descent regression model 
sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_norm, y_train)
w_norm = sgdr.coef_
b_norm = sgdr.intercept_
print(f"sgdr iterations -> {sgdr.n_iter_} ")
print(f"b -> {b_norm} \nw -> {w_norm}")

# make prediction using sgdr
y_sgdr = sgdr.predict(x_norm)

# make prediction using w,b values
y_regression = np.dot(x_norm, w_norm) + b_norm

print(f"y_sgdr -> {y_sgdr} \ny_regression -> {y_regression} \ny_train -> {y_train}")
