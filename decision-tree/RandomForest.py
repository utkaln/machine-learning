import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print(f"=============================================\nRandom Forest Model Starts\n=============================================")

# Create Decision Tree - Steps
# 1 - One hot encoding of categorical data
# 2 - Remove target column from the feature list
# 3 - Split Dataset to Training and Test Dataset


# Dataset reference: Kaggle - Heart failure prediction
# 11 features: age, sex, chest pain type, resting bp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, stslope, heart disease
# Target is heart disease
df = pd.read_csv("./data/heart.csv")
# print(df.head())


# ============================================
# Step 1: One-hot encoding of categorical data
# ============================================
# Categorical data: sex, chest pain type, resting ecg, exercise angina, st slope

category_var = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target_column = 'HeartDisease'
# pandas method to replace with one-hot encoded ones
df = pd.get_dummies(data=df, prefix= category_var, columns= category_var)
# print(df.head())

# ============================================
# Step 2:Remove target value from feature list
# ============================================

features = [x for x in df.columns if x not in target_column]
# print(len(features))

# ============================================
# Step 3:Split dataset to training, validation 
# ============================================
# Pass a random value to be able to reproduce the scenario in future
RANDOM_STATE = 55
x_train, x_val, y_train, y_val = train_test_split(df[features], df[target_column], train_size= 0.8, random_state=RANDOM_STATE)
print(f"train samples: {len(x_train)}\ntest samples: {len(x_val)}")
print(f"target proportion on training set: {sum(y_train)/ len(y_train):.4f}")


# ============================================
# Step 4A: Build Model - Decision Tree
# ============================================
# Use few selected hyperparameters - minimum sample split, max depth etc
# min number of samples to split an internal node - choose higher val to reduce overfitting
min_sample_split = [2, 10, 30, 50, 100, 200, 300, 700]

# maximum depth of the tree - choose lower val to reduce overfitting
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]
# None = No depth limit


# Experiment with controlling different minimum samples to split values
accuracy_list_train = []
accuracy_list_val = []

for min_split in min_sample_split:
    # Define model and fit function at once
    model = DecisionTreeClassifier(min_samples_split=min_split, random_state=RANDOM_STATE).fit(x_train, y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train, y_train)
    accuracy_list_train.append(accuracy_train)
    accuracy_val = accuracy_score(prediction_val, y_val)
    accuracy_list_val.append(accuracy_val)


print(f"accuracy_list_train -> {accuracy_list_train}")
print(f"accuracy_list_val --> {accuracy_list_val}")


# Experiment with controlling different max depth values
accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
    # Define model and fit function at once
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(x_train, y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train, y_train)
    accuracy_list_train.append(accuracy_train)
    accuracy_val = accuracy_score(prediction_val, y_val)
    accuracy_list_val.append(accuracy_val)


print(f"accuracy_list_train -> {accuracy_list_train}")
print(f"accuracy_list_val --> {accuracy_list_val}")




# For optimal target choose max_depth = 4, min_split = 50 for this dataset

decision_tree_model = DecisionTreeClassifier(max_depth=4, min_samples_split=50 ,random_state=RANDOM_STATE).fit(x_train, y_train)
prediction_train = model.predict(x_train)
prediction_val = model.predict(x_val)
accuracy_train = accuracy_score(prediction_train, y_train)
accuracy_val = accuracy_score(prediction_val, y_val)

print(f"accuracy train: {accuracy_train:.2f}")
print(f"accuracy validation: {accuracy_val:.2f}")



# ============================================
# Step 4B: Build Model - Random Forest (Ensemble)
# ============================================
# Random forest uses all the hyperparameters from Decision Tree described above
# 1. Minimum sample split 2. max depth
# Additional hyperparameter used is Number of Trees

print(f"=============================================\nRandom Forest Model Starts\n=============================================")
# min number of samples to split an internal node - choose higher val to reduce overfitting
min_sample_split = [2, 10, 30, 50, 100, 200, 300, 700]

# maximum depth of the tree - choose lower val to reduce overfitting
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]
# None = No depth limit

# n_estimators is number of trees in the forest
n_estimators_list = [10, 50, 100, 500]

accuracy_list_train = []
accuracy_list_val = []

# First experiment with various min sample split values
for min_split in min_sample_split:
    # Define model and fit function at once
    model = RandomForestClassifier(min_samples_split=min_split, random_state=RANDOM_STATE).fit(x_train, y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train, y_train)
    accuracy_list_train.append(accuracy_train)
    accuracy_val = accuracy_score(prediction_val, y_val)
    accuracy_list_val.append(accuracy_val)


print(f"min_split accuracy_list_train -> {accuracy_list_train}")
print(f"min_split accuracy_list_val --> {accuracy_list_val}")



# Experiment with controlling different max depth values
accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
    # Define model and fit function at once
    model = RandomForestClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(x_train, y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train, y_train)
    accuracy_list_train.append(accuracy_train)
    accuracy_val = accuracy_score(prediction_val, y_val)
    accuracy_list_val.append(accuracy_val)


print(f"max_depth accuracy_list_train -> {accuracy_list_train}")
print(f"max_depth accuracy_list_val --> {accuracy_list_val}")


# Experiment with controlling different n_estimators value
accuracy_list_train = []
accuracy_list_val = []

for n_estimators in n_estimators_list:
    # Define model and fit function at once
    model = RandomForestClassifier(n_estimators = n_estimators, random_state=RANDOM_STATE).fit(x_train, y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train, y_train)
    accuracy_list_train.append(accuracy_train)
    accuracy_val = accuracy_score(prediction_val, y_val)
    accuracy_list_val.append(accuracy_val)


print(f"n_estimators accuracy_list_train -> {accuracy_list_train}")
print(f"n_estimators accuracy_list_val --> {accuracy_list_val}")


# For this dataset Choose optimal values as max_depth = 16, min_samples_split = 10, n_estimators = 100

random_forest_model = RandomForestClassifier(
    n_estimators = 100, 
    max_depth=16,  
    min_samples_split=10, 
    random_state=RANDOM_STATE).fit(x_train, y_train)

prediction_train = model.predict(x_train)
prediction_val = model.predict(x_val)
accuracy_train = accuracy_score(prediction_train, y_train)
accuracy_val = accuracy_score(prediction_val, y_val)


print(f"optimal accuracy_train -> {accuracy_train:.4f}")
print(f"optimal accuracy_val --> {accuracy_val:.4f}")