import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset
x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3,0.5], [2,2], [1,2.5]])
y_train = np.array([0,0,0,1,1,1])

# fit to log regression model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# make prediction
y_pred = lr_model.predict(x_train)
print(f"Predicted val of y : {y_pred}")

# calculate accuracy of prediction
print(f"accuracy of prediction: {lr_model.score(x_train,y_train)}")

