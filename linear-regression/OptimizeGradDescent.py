import numpy as np
import matplotlib.pyplot as plt
from MultiLinearRegression import loadHouseData, gradientDescent

# load house data from previous example

x_train, y_train = loadHouseData()
x_features = ['size', 'bedrooms', 'floors', 'age']
w_init = np.array([0.0, 0.0, 0.0, 0.0])
b_init = 0.0

# plot data the way it is found (no feature scale yet)
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

# Set alpha values to see how cost function looks like
iterations = 10
alpha = 9.9e-7
w_final, b_final, cost_history = gradientDescent(x_train, y_train, w_init, b_init, alpha, iterations)
print(f"w -> {w_final} \nb -> {b_final} \ncost_history -> {cost_history} " )

# Progressively try with a lower value of alpha
iterations = 10
alpha = 1e-7
w_final, b_final, cost_history = gradientDescent(x_train, y_train, w_init, b_init, alpha, iterations)
print(f"w -> {w_final} \nb -> {b_final} \ncost_history -> {cost_history} " )
