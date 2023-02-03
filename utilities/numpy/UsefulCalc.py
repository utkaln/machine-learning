import numpy as np
from sympy import  symbols, diff


# Find exponential value where e ~ 2.718

input_arr = np.array([1,2,3,4])
exp_arr = np.exp(input_arr)
print(f"exp array -> {exp_arr}")

# Derivative calculation
J, w = symbols('J,w')
J = w**2 # Plug in different J values here wrt to w
dJ = diff(J,w)

print(f"Differential of J: {J} is dj: {dJ}")

dJ_val = dJ.subs([(w,2)])
print(f"Differential value at J: {J} is : {dJ_val}")


A, b  = symbols('A,b')
A = 1/b # Plug in different J values here wrt to w
dA = diff(A,b)

print(f"Differential of A: {A} is dA: {dA}")


## Softmax calculation
def compute_softmax(z):
    a = np.empty(0)
    denom = 0.
    for i in range (len(z)):
        denom += np.exp(z[i])
    for j in range (len(z)):
        a = np.append(a,np.exp(z[j])/denom)
    return a

print(f"softmax output --> {compute_softmax(np.array([1.,2.,3.,4.]))}")