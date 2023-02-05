import numpy as np    
import time

# Routines to allocate memory and fill array with values . The shape is sent as input param 4
a = np.zeros(4)
print(f" a = {a} and data type = {a.dtype}")

b = np.zeros((4,))
print(f" b = {b} and data type = {b.dtype}")


c = np.random.random_sample(4)
print(f" c = {c} and data type = {c.dtype}")

# allocate memory and fill array with values but do not accept shape as input param
d = np.arange(4.)
print(f" d = {d} and data type = {d.dtype}")

e = np.random.rand(4)
print(f" e = {e} and data type = {e.dtype}")

# =====================================================
# SLICING
# Slicing creates array of indices using set of three values --> (start:stop:step)
f = np.arange(10)
f1 = f[2:8:2]
print(f"f1 = {f1}")

# math operations
f2 = np.sum(f)
print(f"f2 = {f2}")

f3 = np.mean(f)
print(f"f3 = {f3}")

# dot product
g = np.arange(10,20)
fg = np.dot(f,g)
print(f"dot product -> {fg}")

# =========================================================
# VECTOR OPERATIONS

# Create a matrix of 20 elements in 10 columns
# -1 allows to auto calculate number of rows
h  = np.arange(20).reshape(-1, 10)
print(f"h = \n{h}")

# Slicing advanced
#access 5 consecutive elements (start:stop:step)
print("h[0, 2:7:1] = ", h[0, 2:7:1], ",  h[0, 2:7:1].shape =", h[0, 2:7:1].shape, "h 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("h[:, 2:7:1] = \n", h[:, 2:7:1], ",  h[:, 2:7:1].shape =", h[:, 2:7:1].shape, "h 2-D array")

# access all elements
print("h[:,:] = \n", h[:,:], ",  h[:,:].shape =", h[:,:].shape)

# access all elements in one row (very common usage)
print("h[1,:] = ", h[1,:], ",  h[1,:].shape =", h[1,:].shape, "h 1-D array")
# first row
print("h[0]   = ", h[0],   ",  h[0].shape   =", h[0].shape, "h 1-D array")


