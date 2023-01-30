import numpy as np

# Sample Matrix data
a = np.array([[1, -1, 0.1],[2, -2, 0.2]])
b = np.array([[3, 5, 7, 9], [4, 6, 8, 0]])

# Count of columns of 1st matrix must match with count of rows of 2nd matrix
aT = np.transpose(a)

print(f"Input matrix a: \n{a}\naT: \n{aT}\nb: \n{b}")

result = np.matmul(aT,b)

print(f"Matrix Multiplication Result:\n{result}")