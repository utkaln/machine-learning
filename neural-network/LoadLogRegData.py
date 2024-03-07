
import numpy as np
def loadData():
  X = np.array([[185.32,  12.69], [259.92, 11.87], [231.01,  14.41], [175.37,  11.72], [187.12,  14.13],[225.91,  12.1 ],[208.41,  14.18], [207.08,  14.03],[280.6,   14.23], [202.87,  12.25]])
  Y = np.array([[1.],[0.], [0.], [0.], [1.], [1.],[0.],[0.],[0.],[1.]])
  return X,Y

def sigmoid(z):
  return 1/(1+np.exp(-z))

def decisions(prediction):
  decision = np.zeros_like(prediction)
  for i in range(len(prediction)):
    if prediction[i] >= 0.5:
        decision[i] = 1
    else:
        decision[i] = 0
  return(decision)


