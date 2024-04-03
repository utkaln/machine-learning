import numpy as np
from sklearn.linear_model import LogisticRegression

# Load Data from csv file
arr = np.loadtxt("./data/kaggle-data-heart-log-reg.csv",
                 delimiter=",", dtype=str)
# Clean the Data
# Remove the column row
arr = arr[1:,:]

# Replace missing values or NA values
arr[arr == 'NA'] = np.nan
print(arr.dtype)
# Calculate mean value for each column ignoring NAN
mean_val = np.nanmean(arr, axis=0)
# Replace NaN values with mean value in columns
# imputed_data = np.where(np.isnan(arr), 0, arr)
imputed_data = arr

print(imputed_data)
rows, columns = imputed_data.shape
print(f"rows: {rows}, columns: {columns}")
X = imputed_data[:, :15]
y = imputed_data[:, 15:]
print(f"X: {X.shape}, y: {y.shape}")

log_reg_model = LogisticRegression()
log_reg_model.fit(X,y)
y_prediction = log_reg_model.predict(X)
prediction_accuracy = log_reg_model.score(X,y) 

print(f"Prediction Accuracy Score: {prediction_accuracy}")

