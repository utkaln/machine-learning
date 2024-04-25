from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge



# Split data to training 60%, validation 20%, test 20%
# Accepts 2D array (matrix) 
def split_data(x,y):
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.40, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=1)

    # Delete temporary variables
    del x_temp, y_temp  

    return [x_train, y_train, x_cv, y_cv, x_test, y_test]


# Build model utility
# define how to build models
def build_nn_models():
    tf.random.set_seed(20)
    model_1 = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name='model_1_classification'
    )

    model_2 = Sequential(
        [
            Dense(20, activation='relu'),
            Dense(12, activation='relu'),
            Dense(12, activation='relu'),
            Dense(20, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name='model_2_classification'
    )

    model_3 = Sequential(
        [
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(12, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name='model_3_classification'
    )

    return [model_1, model_2, model_3]

def train_polynomial_model(model, x_train, y_train, x_cv, y_cv, max_degree, baseline=None):
    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    degrees = range(1,max_degree+1)

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for degree in degrees:
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model.fit(X_train_mapped_scaled, y_train )
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features 
        poly = PolynomialFeatures(degree,include_bias=False)

        # scale cross validation data
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute cross validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)
    return [train_mses, cv_mses]


def train_with_regularization(regularization_params, x_train, y_train, x_cv, y_cv, degree, baseline):
    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    for reg_param in regularization_params:
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Apply Ridge model of scikit, that allows linear regression with Regularization param
        # Create and train the model
        model = Ridge(alpha=reg_param)
        model.fit(X_train_mapped_scaled, y_train )
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features 
        poly = PolynomialFeatures(degree,include_bias=False)

        # scale cross validation data
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute cross validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)
    return [train_mses, cv_mses]
