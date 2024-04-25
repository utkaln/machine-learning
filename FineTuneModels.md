# Finetuning Models
- To make a model better one or more of the following usual choices are taken as follows: 
    - Get more Training Data
    - Try smaller set of features
    - Try adding more number of features 
    - Try adding Polynomial features
    - Try decreasing regularization factor
    - Try increasing regularization factor


### Baseline Performance
- Baseline Performance: Benchmarking data based on human computed value, or any other reliable source
- Baseline performance helps in setting realistic expectation about error rate. For example if human eyes have error rate of 10% then, error rate of the model at 10% is not performance issue of the model

## Techniques of Finetuning
- Evaluate Model
- Diagnose Bias and Variance
- Controlling Regularization Factor
- Understand Learning Curve


### Evaluate Model
- Split Data to Training Data, Cross Validation Data and Test Data
- Find Error rate with different Models and find how data performs on different Models
- Usually a model with increasing polynomial features trend towards overfitting. Which means - Traing Err: goes high to low | CV Error : goes in U Shape, High to low to High

### Diagnose Bias and Variance
- Bias is Underfit and Variance is overfit
- In case of High Bias Training Error : High and Validation Error : High 
- In case of High Variance Training Error : Low and Validation Error : High 

### Controlling Regularization Factor
- In Linear Regression a regularization term is added at the end of the equation to influence the curve
- If the regularization term is chosen very high, then it suppresses the effects of features and weights and makes the line straight causing High Bias problem
- If the regularization term is chosen very low (close to zero) thus the curve ends up overfitting and causing high Variance problem
- Error rate on Cross Validation guides towards the appropriate value of Regularization Term
- Start with a low value of Regularization Term so that the cross validation error indicates about a overfitting problem. Gradually increase Regularization Term value and see the CV error going down

### Understand Learning Curve
- Learning curve is the projection of Training Error and Validation Error against volume of training data set
- Training Error initially increases as number of training data increase and then reaches a plateau. 
- Cross Validation Error reduces as the number of training data increase and then reaches a plateau. 
- In case of High Bias (Under fit) - Training Error increases rapidly as number of data increase and then flattens. Similarly Validation Error decreases rapidly and then flattens. Hence the conclusion is that in case of High Bias increasing number of data does not help much with reductin of error
- In case of High Variance (Overfit) - Training Error Continues to grow up with the increase in number of training data, but the Validation Error does not increase, with a substantial difference between Training Error and Validation error. Hence increasing number of training data eventually brings the gap closer

## Summary of Fine Tuning

| Model Situation | Fix By |
| --- | --- | 
| Get more Training Data | Fixes High Variance | 
| Try smaller set of Features | Fixes High Variance | 
| Try getting additional Features | Fixes High Bias | 
| Try adding Polynomial Features | Fixes High Bias | 
| Try decreasing Regularization Factor | Fixes High Bias | 
| Try decreasing Regularization Factor | Fixes High Variance | 

## Fix Bias and Variance in Neural Network
- Usually large neural networks take care of bias naturally
- To take care of variance add more training data
- However, Neural network performance can be fine tuned by using Regularization factor


## Problem with Skewed Data
- If a dataset has very skewed data, it is important to understand the impact before infering.
- Example if the data it self has 1% of True and 99% false, then the error rate is not a good valuation approach of the model
- Using Confusion Matrix two Important factors can be found - **Precision** and **Recall**

### Precision
- Fraction of True Positive over Predicted Positives
- Higher Precision indicates better model quality


### Recall
- Fraction of True Positive over Actual Positives
- Higher Recall indicates better model quality

### F1 Score : Tradeoffs - Precision and Recall
- However having both Precision and Recall high is not always possible, hence a Trade off is done
- In case of classification, increasing the Threshold value increases Precsion but decreases Recall
- Hence using F1 score it can be found out which one has the best combination of Precision and Recall
- `F1 Score = 1 / [(1/P + 1/R)* 1/2]` or `2*PR/(P+R)`
- F1 score goes closer to the lower value between  P and R




