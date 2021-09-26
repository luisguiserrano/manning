# -*- coding: utf-8 -*-

import Coding_logistic_regression
import numpy
import matplotlib.pyplot as plt

features = numpy.array([
    [0,2],
    [1,0],[1,1],[1,2],[1,3],
    [2,2],[2,3],
    [3,2]
    ])

labels = numpy.array([0,0,0,0,1,1,1,1])

data = numpy.column_stack((features, labels))



x = Coding_logistic_regression.logistic_regression_algorithm(
    features, labels, learning_rate = 0.1, num_epochs = 500)

Coding_logistic_regression.plot_scatter(data[labels == 0,0], data[labels == 0,1], marker = '^')
Coding_logistic_regression.plot_scatter(data[labels == 1,0], data[labels == 1,1], marker = 's')
plt.legend(["Happy", "Sad"])
#Coding_logistic_regression.draw_line(-1,3.5, ending = 8)
Coding_logistic_regression.draw_line(-x[0][0]/x[0][1], -x[1]/x[0][1], ending=3)
plt.show()

# # Logistic regression using Turi Create


# import turicreate as tc

# data = tc.SFrame({'x1': features[:,0], 'x2': features[:,1], 'y': labels})
# data



# classifier = tc.logistic_classifier.create(data,
#                                             features = ['x1', 'x2'],
#                                             target = 'y',
#                                             validation_set=None)


# print("Model coefficients", classifier.coefficients)


# intercept, w1, w2 = classifier.coefficients['value']

# plot_scatter(features, labels)
# draw_line(-w1/w2, -intercept/w2)