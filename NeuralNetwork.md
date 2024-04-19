# Neural Network
- Neural Network is a cluster of logistic regression performed in sequence
- Each Previous Layer's output is considered as input variable to the next layer
- Each node of computing is called a neuron, where a logistic regression equation is solved using sigmoid function

## Tensorflow
- Tensorflow and Keras are machine learning packages widely used in neural networks
- Tensorflow always accepts data as 2 dimensional matrix, hence declare the values in numpy with 2 square brackets e.g. of 2x3 matrix :  `np.array([[1,2,3],[4,5,6]])`

## Fundamental Steps of Neural Network using Tensorflow
- This is very similar to that of the Logistic regression Steps:
    - `z = np.dot(w,x) + b`
    - `sigmoid(z) = 1 / (1 + np.exp(-z))`
    - `Loss = -y * np.log(sigmoid(z))`
    - `Cost = Average of Loss`
    - Train on Data = `w = w - alpha * dj_dw`

1. Build Inference by Creating a Model
```python
model = Sequential(
    [
      ...
    ]
)
```
2. Define Type of Function (Regression, Sigmoid etc.)
```python
model = Sequential(
    [
      tf.keras.Input(shape=(2,)),
      Dense (3, activation='sigmoid', name= 'layer1'),
      Dense (1, activation='sigmoid', name= 'layer2')
    ]
)
```
3. Compile the model by Defining Loss Function
```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
```
4. Train the Model with Prediction, Known Output value and the Loss Function Input (Run Gradient Descnt)
```python
model.fit(Xt, Yt, epochs=10,)
```

### Activation Function
- If the final output is Classification then choose Activation function as `sigmoid`
- If the final output is Continuous Number then choose Activation Function as `linear`
- All the inner layers are better suited for `ReLU`


## Multiclass Classification using Softmax
- This type of classification deals with more than two discrete output. Hence instead of just true/false output it can take more type of expected output values
- Each output has a threshold probability
- The calculation of Multiclass Classification is done using Softmax Regression Equation. This is a generalization of Logistic Regression model
- Probability of a1 : 
    - Calculate `z1 = (w1 * x1) + b1`
    - Probability `a1 = e**z1 / (e**z1 + e**z2 + e**z3 + e**z4 + ...)`

### Cost Function of Softmax
- `loss = - log(a1)`
- In summary softmax is a calculation of probability. Important to remember that each neuron's probability calculation is dependent on all the individual regression values


## How to Finetune a Model
- Split the Data set to Three parts
    - Training Set (60%)
    - Validation Set (20%)
    - Testing Set (20%)
- Important to remember that while feature scaling, keep the mean and standard deviation value consistently used after once calculating for the training set
- Do not compute mean and standard deviation once again for validation or test set




