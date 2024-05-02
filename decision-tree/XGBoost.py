from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd

# XG Boost uses same hyper parameters as that of the Decision Tree and Random forests
# hyperparameters used - minimum data for split, max depth, number of trees (n estimates)
# Additionally it uses learning rate as a hyperparameter that drives size of the step on Gradient Descent method that XGBoost uses to minimize error on each training step
# Evaluation Steps:
# 1. On each iteration measure cost on validation dataset, Hence must have 3 segments of dataset - Training Data, Validation Data and Test Data
# 2. Once cost stops decreasing, it stops training


# Pass a random value to be able to reproduce the scenario in future
RANDOM_STATE = 55

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
# Step 3:Split dataset to training, validation, test 
# ============================================

x_train, x_test, y_train, y_test = train_test_split(df[features], df[target_var], train_size= 0.8, random_state=RANDOM_STATE)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.8,random_state=RANDOM_STATE)


# ============================================
# Step 4: Build Model - XGBoost using scikit
# ============================================

# XGboost uses valuation dataset to measure error rate, eval_set is the parameter for that
# early stopping rounds is a value, where the model stops computing after the best iteration
# Important: Though n_estimators or number of trees is provided as a big number, the other hyperparameter named early_stopping_rounds plays a role - As soon as a best value is found, the model tries only those many counts defined in early stopping round to see if there is any further improvement. If not it exits.
# In the following case n_estimators = 500, however, when the model is run the best value comes at 16. So the model runs 10 after that and exits at 26th iteration

xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1, verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = 10)
prediction_train = xgb_model.predict(x_train)
prediction_val = xgb_model.predict(x_val)
accuracy_train = accuracy_score(prediction_train, y_train)
accuracy_val = accuracy_score(prediction_val, y_val)

print(f"Best Iteration: {xgb_model.best_iteration}")
print(f"Training Accuracy: {accuracy_train}")
print(f"Validation Accuracy: {accuracy_val}")