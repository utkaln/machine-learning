import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Pass a random value to be able to reproduce the scenario in future
RANDOM_STATE = 55

# Dataset reference: KAggle - Heart failure prediction
# 11 features: age, sex, chest pain type, resting bp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, stslope, heart disease
# Target is heart disease
df = pd.read_csv("./data/heart.csv")
# print(df.head())


# ============================================
# Step 1: One-hot encoding of categorical data
# ============================================
# Categorical data: sex, chest pain type, resting ecg, exercise angina, st slope

category_var = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target_var = 'HeartDisease'

# pandas method to replace with one-hot encoded ones
df = pd.get_dummies(data=df, prefix= category_var, columns= category_var)
# print(df.head())

# ============================================
# Step 2:Remove target value from feature list
# ============================================

features = [x for x in df.columns if x not in target_var]
# print(len(features))

# ============================================
# Step 3:Split dataset to training, validation 
# ============================================
x_train, x_val, y_train, y_val = train_test_split(df[features], df[target_var], train_size= 0.8, random_state=RANDOM_STATE)
print(f"train samples: {len(x_train)}\ntest samples: {len(x_val)}")
print(f"target proportion on training set: {sum(y_train)/ len(y_train):.4f}")


# ============================================
# Step 4A: Build Model - Decision Tree
# ============================================
# min number of samples to split an internal node - choose higher val to reduce overfitting
min_split_list = [2, 10, 30, 50, 100, 200, 300, 700]

# maximum depth of the tree - choose lower val to reduce overfitting
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]


# Experiment with controlling different minimum samples to split values
accuracy_list_train = []
accuracy_list_val = []

for min_split in min_split_list:
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


plt.xlabel('min samples split')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(min_split_list)), labels=min_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['train', 'validation'])


# Experiment with controlling different minimum samples to split values
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


plt.xlabel('min samples split')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(min_split_list)), labels=min_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['train', 'validation'])


# For optimal target choose max_depth = 3, min_split = 50 for this dataset

decision_tree_model = DecisionTreeClassifier(max_depth=3, min_samples_split=50 ,random_state=RANDOM_STATE).fit(x_train, y_train)
prediction_train = model.predict(x_train)
prediction_val = model.predict(x_val)
accuracy_train = accuracy_score(prediction_train, y_train)
accuracy_val = accuracy_score(prediction_val, y_val)

print(f"accuracy train: {accuracy_train:.4f}")
print(f"accuracy validation: {accuracy_val:.4f}")



# ============================================
# Step 4B: Build Model - Random Forest
# ============================================

