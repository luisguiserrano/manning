#!/usr/bin/env python
# coding: utf-8
"""
Sentiment analysis with Logistic Regression

Code base is almost exactly the same as perceptron,
since the only difference is we're moving to cts space.
"""

import matplotlib.pyplot as plt
import numpy
import tqdm

# Helpers (Plotting) =========================================

def plot_scatter(x_iterable, y_iterable, x_label = "", y_label = "",  legend = None, **kwargs):
    x_array = numpy.array(x_iterable)
    y_array = numpy.array(y_iterable)
    plt.xlabel(x_label)
    plt.xlabel(y_label)
    if legend is not None:
        plt.legend(legend)
    plt.scatter(x_array, y_array, **kwargs)

def draw_line(slope, y_intercept, starting=0, ending=8, **kwargs):
    x = numpy.linspace(starting, ending, 1000)
    plt.plot(x, y_intercept + slope*x, **kwargs)


# ### Logistic regression

def sigmoid(x):
    """
    Note: The book's version is 1/(1+exp(-x)).
    Our version exp(x) / (1 + exp(x)) is an equivalent expression.
    But this behaves better with small floating point numbers.
    """
    exponential = numpy.exp(x)
    denom = numpy.add(1, exponential)

    result = numpy.divide(exponential, denom)
    result = numpy.minimum(result, 0.9999)  # Set upper bound
    result = numpy.maximum(result, 0.0001)  # Set lower bound
    
    return result

def calculate_score(array_weights, bias, array_feature):
    inner_product = numpy.dot(array_weights, array_feature)

    return numpy.add(inner_product, bias)

def calculate_prediction(array_weights, bias, array_feature):
    score = calculate_score(array_weights, bias, array_feature)

    return sigmoid(score)

def log_loss(array_weights, bias, array_feature, label):
    """
    error = -label*log(pred) - (1-label)*log(1-pred)
    """
    prediction = calculate_prediction(array_weights, bias, array_feature)

    term1 = numpy.multiply(
        label,
        numpy.log(prediction)
        )
    term2 = numpy.multiply(
        numpy.subtract(1, label),
        numpy.log(numpy.subtract(1, prediction))
        )

    return -1*numpy.add(term1, term2)

def total_log_loss(array_weights, bias, array_features, array_labels):
    """
    I'm pretty sure I can pass the whole vector
    and be okay avoiding the for-loop
    """
    total_error = 0
    for feature, label in zip(array_features, array_labels):
        total_error += log_loss(array_weights, bias, feature, label)
    return total_error

# Alternate way of writing log-loss

def soft_relu(x):
    inside = numpy.add(1, numpy.exp(x))
    return numpy.log(inside)

def log_loss_alternate(array_weights, bias, array_feature, label):
    """
    Observe we essenially calcluate the score twice,
    but we need the probability.

    error = soft_relu( (label-prediction)*score )
    """
    prediction = calculate_prediction(array_weights, bias, array_feature)
    score = calculate_score(array_weights, bias, array_feature)
    residual = numpy.subtract(prediction,label)
    x = numpy.multiply(residual, score)

    return soft_relu(x)

def total_log_loss_alternate(array_weights, bias, array_features, array_labels):
    total_error = 0
    for feature, label in zip(array_features, array_labels):
        total_error += log_loss_alternate(array_weights, bias, feature, label)
    return total_error

def logistic_trick(array_weights, bias, array_feature, label, learning_rate = 0.01):
    """
    Update the weights using the calculated prediction

    new weight_vector = weight_vector + (label-prediction)*feature_vector*learning_rate
    new bias = bias + (label-prediction)
    """
    prediction = calculate_prediction(
        array_weights, bias, array_feature)
    residual = numpy.subtract(label, prediction)
    bias_update = numpy.multiply(residual, learning_rate)
    # Update weights and bias
    array_weights = numpy.add(
        array_weights,
        numpy.multiply(bias_update, array_feature))
    bias = numpy.add(bias, bias_update)

    return array_weights, bias

def logistic_regression_algorithm(array_features, array_labels, learning_rate = 0.01, num_epochs = 1000):
    """
    Loop breaks when converges or if num_epochs is reached

    Stores the best weights and bias in case of non-convergence
    """
    assert array_features.shape[0] == array_labels.shape[0]

    array_weights = numpy.ones(shape = array_features.shape[1])
    bias = 0.0
    best_weights = None
    best_bias = None

    # base case
    count = 0
    error = total_log_loss(
        array_weights, bias, array_features, array_labels)
    iter_errors = [error]

    progress_bar = tqdm.tqdm(total = num_epochs)
    while (error >= 1e-16) and (count <= num_epochs):

        error = total_log_loss(
            array_weights, bias, array_features, array_labels)

        # Identifies best weights
        if error < min(iter_errors):
            best_weights = array_weights
            best_bias = bias
        iter_errors.append(error)

        # Updates weights & bias
        index = numpy.random.randint(0, array_features.shape[0] - 1)
        array_weights, bias = logistic_trick(
            array_weights,
            bias,
            array_features[index],
            array_labels[index],
            learning_rate)

        count +=1

    progress_bar.close()

    # Plotting error
    plot_scatter(
        range(len(iter_errors)),
        iter_errors,
        x_label = 'epochs',
        y_label = 'error')
    plt.title("Log 'Loss' Error per Iteration")
    plt.show()

    return best_weights, best_bias







