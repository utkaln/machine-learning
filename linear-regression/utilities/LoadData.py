import numpy as np
# Example Test data 
# Input features: Size of House, Bedroom count, Floor Count, Home Age
# Target Value: Price of house

def loadHouseData():
  x = np.array([[2104, 5, 1, 45],[1416, 3, 2, 40],[852, 2, 1, 35]])
  y = np.array([460, 232, 178])
  return x, y

# Starting value of W and b that usually gets determined later
# For practice at the moment this is given

def preInitWb():
  b_init = 785.1811367994083
  w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
  return b_init, w_init
