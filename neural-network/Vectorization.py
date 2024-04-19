import numpy as np
from LoadLogRegData import sigmoid

X = np.array([[200,17]]) # 2 dimensional array
W = np.array([[1,-3,5],[-2,4,6]])
B = np.array([[-1,1,2]])

def dense(X_in,W,B):
    z = np.matmul(X_in,W) + B
    return sigmoid(z)

