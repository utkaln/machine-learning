import numpy as np
from MultiLinearRegression import loadHouseData, gradientDescent
from utilities.RegressionUtils import predict_regression
import time
x_train, y_train = loadHouseData()

def zscore_normalize_features(x):
    mean_val = np.mean(x,axis=0)
    std_dev = np.std(x,axis=0)
    z_norm = (x - mean_val)/std_dev
    return z_norm

x_norm = zscore_normalize_features(x_train)
print(f'x_train -> {x_train}\nx_norm -> {x_norm}')

w_init = np.array([0.0, 0.0, 0.0, 0.0])
b_init = 0.0
alpha = 1.0e-1
iterations = 1000
start_time = time.time()
w_norm, b_norm, cost_history = gradientDescent(x_norm, y_train, w_init, b_init, alpha, iterations)
m = x_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    print(f'y_norm[{i}] -> {predict_regression(x_norm[i],w_norm,b_norm)}') 

print(f'y_train -> {y_train}')

# try with a new data
x_house = np.array([1200, 3, 1, 40])
x_house_norm = zscore_normalize_features(x_house)
print(f'y_house -> {predict_regression(x_house_norm,w_norm,b_norm)}') 



