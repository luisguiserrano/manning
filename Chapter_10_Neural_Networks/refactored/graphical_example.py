
# # A graphical example

import pandas
import numpy
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

import _plotting

# Setting random seeds to get reproducible results
numpy.random.seed(0)
# tensorflow.random.set_seed(1)

# Read the dataset
data = pandas.read_csv('one_circle.csv', index_col=0)
columns_features = ['x_1', 'x_2']
column_label = 'y'
features = data[columns_features].values
labels = data[column_label].values

_plotting.plot_scatter(data['x_1'][labels == 0], data['x_2'][labels == 0], marker = 's')
_plotting.plot_scatter(data['x_1'][labels == 1], data['x_2'][labels == 1], marker = '^')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(["Happy", "Sad"])
plt.show()

# ### Preprocess

# Categorizing the output
series_labels, labels = pandas.factorize(data[column_label])

### Building and compiling the neural network

# Building the model
model = Sequential()
model.add(Dense(2**7, activation='relu', input_shape=(features.shape[1],)))
model.add(Dropout(.2))
model.add(Dense(2**6, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(len(labels), activation='sigmoid')) # two classes, else softmax

# Compiling the model
model.compile(
    loss = 'sparse_categorical_crossentropy', # two classes, else categorical_crossentropy
    optimizer='adam', 
    metrics=['accuracy']
)
model.summary()

# ### Training the neural network

# Training the model
model.fit(features, series_labels, epochs=100, batch_size=10)

# ### Plotting the results
_plotting.plot_decision_boundary_2D(features, series_labels, model)










