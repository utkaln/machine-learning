import copy, math
import numpy as np

def predict_regression(x,w,b):
    return np.dot(x,w) + b


# Error Prediction function to be used in Gradient Descent calculation
def error_regression(x,y,w,b):
  return (predict_regression (x,w,b) - y)

# Cost Function
def cost_regression(x,y,w,b):
  m = x.shape[0]
  cost = 0.0
  for i in range(m):
    predict_i = predict_regression(x[i], w, b)
    cost = cost + (predict_i - y[i])**2
  cost = cost / (2 * m)
  return cost

