import numpy as np
# z = w.x + b
def z_fn(w,x,b):
    return np.dot(x,w) + b

def sigmoid_fn(z):
  return 1/(1+np.exp(-z))

def log_reg_cost_fn(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        z_i = z_fn(x[i],w, b)
        sigmoid_i  = sigmoid_fn(z_i)
        cost += -y[i]*np.log(sigmoid_i) - (1-y[i])*np.log(1-sigmoid_i)
    cost = cost / (m)
    return cost

# Example of cost calculation
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
y_train = np.array([0, 0, 0, 1, 1, 1]) 
w_temp = np.array([1,1])
b_temp = -3
print(f'Cost Calculation of Log Regression is computed as - > {log_reg_cost_fn(X_train, y_train,w_temp,b_temp)}')
