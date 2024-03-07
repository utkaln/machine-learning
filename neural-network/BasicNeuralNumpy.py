import numpy as np
from LoadLogRegData import loadData, sigmoid, decisions

def baseLogReg(a,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for i in range(units):
        w = W[:,i]
        z = np.dot(w,a) + b[i]
        a_out[i]= sigmoid(z)
    return a_out

def numpyDense(a_in, W,b):
    return baseLogReg(a_in, W, b)

def twoLSequential(x,W1,b1,W2,b2):
    a1 = numpyDense(x,W1,b1)
    a2 = numpyDense(a1,W2,b2)
    return a2

def numPyPredict(X,W1,b1,W2,b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = twoLSequential(X[i],W1,b1,W2,b2)
    return p


# Copy data from previous training
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

# Sample data for prediction
X_tst = np.array([[200,13.9],[200,17]]) 
prediction = numPyPredict(X_tst,W1_tmp,b1_tmp,W2_tmp,b2_tmp)

dec = decisions(prediction)
print(f" decision --> {dec}")

    
    
    