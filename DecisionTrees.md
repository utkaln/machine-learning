# Decision Tree

## Single Decision Tree
- This is a type of classification that deals with splitting the values based on the possible discrete choices at each decision points
- When the value of split finally reaches 100% desired classification, further split is stopped and the bottom values are called leaf node
- It is important to decide on what classification to choose at what level of decision tree to efficiently reach to the output
- Entropy is a measurement of Purity that is required to reach at leaf nodes efficiently. Lower the value more pure it is. In other words higher Information Gain
- The value of Entropy ranges between 0 to 1. When the sample probability is 50%, Entropy is highest at that point.
- Formula to count Entropy -> `H(p1) = -p1 * log2(p1) - (1-p1)*log2(1-p1)`
- The base 2 for log provides a nice smooth curve compared to that of natural exponent 2
- Information Gain is computed as -> `H(p1node) - (wLeft * H(p1Left) + wRight * H(p1Right))`

## Tree Ensemble - Random Forest
- Instead of One decision tree that starts with a selected feature as the decision node, use multiple independent trees to have all possible features as a starting decision node. Then decide on the prediction through majority of predicted values by individual trees

## Tree Ensemble - XGBoost
- Using a learning rate focus on further fine tuning where the accuracy of prediction is less


## Example Exercise
- [Kaggle Dataset - Cardiovascular data](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Convert Categorical data using One hot encode - sex, chest pain type, resting ECG, Exercise Angina, ST Slope

