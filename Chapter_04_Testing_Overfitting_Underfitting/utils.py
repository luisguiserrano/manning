import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_and_plot_polynomial_regression(X, Y, degree):
    """
    Trains a polynomial regression model, returns weights, and plots results.

    Args:
      X: Input features (list or numpy array).
      Y: Input labels (list or numpy array).
      degree: The degree of the polynomial.

    Returns:
      A tuple containing:
        - The weights (coefficients) of the trained model (numpy array).
        - The intercept of the trained model (float).
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, Y)

    # Generate predicted values for plotting the curve
    X_plot = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    Y_plot_poly = model.predict(X_plot_poly)

    # Plot the original points
    plt.scatter(X, Y, color='blue', label='Original Data')

    # Plot the polynomial regression curve
    plt.plot(X_plot, Y_plot_poly, color='red', label=f'Polynomial Regression (degree {degree})')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()
    plt.grid(True)

    # Set plot bounds based on X and Y
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))

    plt.show()

    return model.coef_, model.intercept_

def predict_and_evaluate(model_coefficients, degree, X_train, Y_train, X_test, Y_test):
    """
    Makes predictions using a polynomial regression model, calculates the square loss,
    and plots the training and testing set points with the regression curve.

    Args:
      model_coefficients: The coefficients of the trained polynomial model (tuple of (coef_, intercept_)).
      degree: The degree of the polynomial.
      X_train: The training set features (list or numpy array).
      Y_train: The training set labels (list or numpy array).
      X_test: The test set features (list or numpy array).
      Y_test: The test set labels (list or numpy array).

    Returns:
      The mean squared error (square loss) on the test set.
    """
    X_train = np.array(X_train).reshape(-1, 1)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1, 1)
    Y_test = np.array(Y_test)

    coefs = model_coefficients[0] # model_coefficients is a tuple (coef_, intercept_)
    intercept = model_coefficients[1]

    # Reconstruct the polynomial transformation for prediction
    poly = PolynomialFeatures(degree=degree)
    # Fit on the combined data range to ensure the transformation is consistent
    poly.fit(np.concatenate((X_train, X_test)))

    X_test_poly = poly.transform(X_test)


    # Build the prediction manually using the coefficients and degree.
    # The intercept is the coefficient for the x^0 term
    y_pred = X_test_poly @ coefs.reshape(-1, 1) + intercept

    # Calculate the mean squared error
    mse = mean_squared_error(Y_test, y_pred)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot training points
    plt.scatter(X_train, Y_train, color='blue', label='Training Data')

    # Plot testing points
    plt.scatter(X_test, Y_test, color='orange', marker='^', label='Testing Data')

    # Generate points for plotting the regression curve over the entire range of data
    X_plot = np.linspace(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))), 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)

    # Calculate predictions for the plot
    y_plot_pred = X_plot_poly @ coefs.reshape(-1, 1) + intercept


    # Plot the polynomial regression curve
    # Ensure the plot color is red here
    plt.plot(X_plot, y_plot_pred.flatten(), color='red', label=f'Polynomial Regression (degree {degree})')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Polynomial Regression (Degree {degree}) with Train/Test Data')
    plt.legend()
    plt.grid(True)

    # Set plot bounds based on the range of all data (train + test)
    all_Y = np.concatenate((Y_train, Y_test))
    min_y = np.min(all_Y)
    max_y = np.max(all_Y)

    # Add a little padding to the y-limits
    padding = (max_y - min_y) * 0.1
    plt.xlim(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))))
    plt.ylim(min_y - padding, max_y + padding)


    plt.show()


    return mse

def predict_and_evaluate(model_coefficients, degree, X_train, Y_train, X_test, Y_test):
    """
    Makes predictions using a polynomial regression model, calculates the square loss,
    and plots the training and testing set points with the regression curve.

    Args:
      model_coefficients: The coefficients of the trained polynomial model (tuple of (coef_, intercept_)).
      degree: The degree of the polynomial.
      X_train: The training set features (list or numpy array).
      Y_train: The training set labels (list or numpy array).
      X_test: The test set features (list or numpy array).
      Y_test: The test set labels (list or numpy array).

    Returns:
      The mean squared error (square loss) on the test set.
    """
    X_train = np.array(X_train).reshape(-1, 1)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1, 1)
    Y_test = np.array(Y_test)

    coefs = model_coefficients[0] # model_coefficients is a tuple (coef_, intercept_)
    intercept = model_coefficients[1]

    # Reconstruct the polynomial transformation for prediction
    poly = PolynomialFeatures(degree=degree)
    # Fit on the combined data range to ensure the transformation is consistent
    poly.fit(np.concatenate((X_train, X_test)))

    X_test_poly = poly.transform(X_test)


    # Build the prediction manually using the coefficients and degree.
    # The intercept is the coefficient for the x^0 term
    y_pred = X_test_poly @ coefs.reshape(-1, 1) + intercept

    # Calculate the mean squared error
    mse = mean_squared_error(Y_test, y_pred)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot training points
    plt.scatter(X_train, Y_train, color='blue', label='Training Data')

    # Plot testing points
    plt.scatter(X_test, Y_test, color='orange', marker='^', label='Testing Data')

    # Generate points for plotting the regression curve over the entire range of data
    X_plot = np.linspace(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))), 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)

    # Calculate predictions for the plot
    y_plot_pred = X_plot_poly @ coefs.reshape(-1, 1) + intercept


    # Plot the polynomial regression curve
    # Ensure the plot color is red here
    plt.plot(X_plot, y_plot_pred.flatten(), color='red', label=f'Polynomial Regression (degree {degree})')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Polynomial Regression (Degree {degree}) with Train/Test Data')
    plt.legend()
    plt.grid(True)

    # Set plot bounds based on the range of all data (train + test)
    all_Y = np.concatenate((Y_train, Y_test))
    min_y = np.min(all_Y)
    max_y = np.max(all_Y)

    # Add a little padding to the y-limits
    padding = (max_y - min_y) * 0.1
    plt.xlim(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))))
    plt.ylim(min_y - padding, max_y + padding)


    plt.show()


    return mse

def train_and_plot_regularized_polynomial_regression(X_train, Y_train, X_test, Y_test, degree, regularization_type, alpha=1.0):
    """
    Trains a regularized polynomial regression model (Lasso or Ridge),
    returns the RMSE on the test set, and plots the results.

    Args:
    X_train: Training set features (list or numpy array).
    Y_train: Training set labels (list or numpy array).
    X_test: Test set features (list or numpy array).
    Y_test: Test set labels (list or numpy array).
    degree: The degree of the polynomial.
    regularization_type: Type of regularization ('L1' for Lasso, 'L2' for Ridge).
    alpha: The regularization strength (alpha).

    Returns:
    The Root Mean Squared Error (RMSE) on the test set.
    """
    X_train = np.array(X_train).reshape(-1, 1)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1, 1)
    Y_test = np.array(Y_test)

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    if regularization_type == 'L1':
        model = Lasso(alpha=alpha, max_iter=10000) # Increase max_iter for potential convergence issues
    elif regularization_type == 'L2':
        model = Ridge(alpha=alpha)
    else:
        raise ValueError("regularization_type must be 'L1' or 'L2'")

    model.fit(X_train_poly, Y_train)
    y_pred_test = model.predict(X_test_poly)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred_test))

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot training points
    plt.scatter(X_train, Y_train, color='blue', label='Training Data')

    # Plot testing points
    plt.scatter(X_test, Y_test, color='green', label='Testing Data')

    # Generate points for plotting the regression curve over the entire range of data
    X_plot = np.linspace(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))), 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot_pred = model.predict(X_plot_poly)

    # Plot the regularized polynomial regression curve in red
    plt.plot(X_plot, y_plot_pred, color='red', label=f'Regularized Polynomial Regression (Degree {degree}, {regularization_type}, alpha={alpha:.2f})')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Regularized Polynomial Regression (Degree {degree}) with Train/Test Data')
    plt.legend()
    plt.grid(True)

    # Set plot bounds based on the range of all data (train + test)
    plt.xlim(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))))
    plt.ylim(np.min(np.concatenate((Y_train, Y_test))), np.max(np.concatenate((Y_train, Y_test))))


    plt.show()

    return rmse