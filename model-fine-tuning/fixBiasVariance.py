import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from FineTuneUtils import split_data, train_polynomial_model, train_with_regularization


# Objective: Address High Bias
# Approach 1: Fix High Bias by adding polynomial features
# Approach 2: Get more training data features
# Approach 3: Decrease Regularization Parameter

# Approach 1: Fix High Bias by adding polynomial features
data = np.loadtxt('./data/data_3.csv', delimiter=',')
# Split the inputs and outputs into separate arrays
x = data[:,:-1]
y = data[:,-1]
# Split Data to Train, Validation, Test sets
[x_train, y_train, x_cv, y_cv, x_test, y_test] = split_data(x,y)

# Select the model to be linear regression
model = LinearRegression()
max_degree = 10
baseline = 400
[train_mses, cv_mses] = train_polynomial_model(model,x_train, y_train, x_cv, y_cv,max_degree, baseline)
print(f'First Training MSEs: {np.argmin(train_mses)}: {np.min(train_mses)}\nFirst CV MSEs: {np.argmin(cv_mses)}: {np.min(cv_mses)}')

# Approach 2: Get more training data features
data = np.loadtxt('./data/data_4.csv', delimiter=',')
# Split the inputs and outputs into separate arrays
x = data[:,:-1]
y = data[:,-1]
# Split Data to Train, Validation, Test sets
[x_train, y_train, x_cv, y_cv, x_test, y_test] = split_data(x,y)

# Select the model to be linear regression
model = LinearRegression()
max_degree = 10
baseline = 400
[train_mses, cv_mses] = train_polynomial_model(model,x_train, y_train, x_cv, y_cv,max_degree, baseline)
print(f'Second Training MSEs: {np.argmin(train_mses)}: {np.min(train_mses)}\nSecond CV MSEs: {np.argmin(cv_mses)}: {np.min(cv_mses)}')

# Approach 3: Decrease Regularization Parameter
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]
poly_degree = 4
baseline = 250
[train_mses, cv_mses] = train_with_regularization(reg_params, x_train, y_train, x_cv, y_cv,poly_degree, baseline)
print(f'Third Training MSEs: {np.argmin(train_mses)}: {np.min(train_mses)}\nThird CV MSEs: {np.argmin(cv_mses)}: {np.min(cv_mses)}')


# ================

# Objective: Address High Variance
# Approach 1: Increase Regularization Parameter
# Approach 2: Smaller set of data features
# Approach 3: Get More Training Examples 

# Approach 1: Increase Regularization Parameter
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
poly_degree = 4
baseline = 250
[train_mses, cv_mses] = train_with_regularization(reg_params, x_train, y_train, x_cv, y_cv,poly_degree, baseline)
print(f'Fourth Training MSEs: {np.argmin(train_mses)}: {np.min(train_mses)}\nFourth CV MSEs: {np.argmin(cv_mses)}: {np.min(cv_mses)}')


