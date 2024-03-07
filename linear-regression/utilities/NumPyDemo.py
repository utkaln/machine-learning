import numpy as np    
import time

# Routines to allocate memory and fill array with values . The shape is sent as input param 4
a = np.zeros(4)
print(f"New List with 4 items, all valued 0 ->  a = {a} and data type = {a.dtype}")

b = np.zeros((4,))
print(f" b = {b} and data type = {b.dtype}")


c = np.random.random_sample(4)
print(f"random.random_sample -> c = {c} and data type = {c.dtype}")

# allocate memory and fill array with values but do not accept shape as input param
d = np.arange(4.)
print(f"Start a List from 0->4 :  d = {d} and data type = {d.dtype}")

e = np.random.rand(4)
print(f" random.rand(4) -> e = {e} and data type = {e.dtype}")

# =====================================================
# SLICING
# Slicing creates array of indices using set of three values --> (start:stop:step)
f = np.arange(10)
f1 = f[2:8:2]
print(f"f[2:8:2] -> f1 = {f1}")

# math operations
f2 = np.sum(f)
print(f"sum(f) -> f2 = {f2}")

f3 = np.mean(f)
print(f"mean(f) -> f3 = {f3}")

# dot product
g = np.arange(10,20)
fg = np.dot(f,g)
print(f"dot product -> {fg}")

# =========================================================
# VECTOR OPERATIONS

# Create a matrix of 20 elements in 10 columns
# -1 allows to auto calculate number of rows
h  = np.arange(20).reshape(-1, 10)
print(f"Create Rows of Data from Flat List -> h = \n{h}")

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


# VECTOR DOT PRODUCT
# First calculate time taken for iterative multiplication
def suboptimal_dot(a,b):
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
np.random.seed(1)
a = np.random.rand(1000000)
b = np.random.rand(1000000)
# sub_start_time = time.time()
# print(f'{suboptimal_dot(a,b)}')
# sub_end_time = time.time()
# print(f'time of sub Optimal dot -> {sub_end_time - sub_start_time}')
dot_start_time = time.time()
print(f'{np.dot(a,b)}')
dot_end_time = time.time()
print(f'Time of np.dot -> {dot_end_time - dot_start_time}')

# Matrix Creation

print(f'2x3 matrix with zeroes -> \n{np.zeros((2,3))}')

print(f'4x3 matrix with random numbers -> \n{np.random.random_sample((4,3))}')

