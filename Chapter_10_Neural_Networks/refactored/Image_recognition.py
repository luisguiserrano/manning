
# # Building an image recognition neural network


import numpy
import pandas
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU

# Setting random seeds to get reproducible results
numpy.random.seed(0)
tensorflow.random.set_seed(1)

# ### Importing and reading the dataset
# The built-in dataset is already balanced, so we don't need to worry about
# stratifying classes to ensure equal representation
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
print("Training set size:", len(x_train))
print("Testing set size:", len(x_test))


plt.imshow(x_train[5], cmap='Greys')
print("The label is", y_train[5])

fig = plt.figure(figsize=(20,20))
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='Greys')
    ax.set_title('Label:' + str(y_train[i]))


# ### Pre-processing the data


# Reshaping the features.
# In the reshape function we use the -1 as a placeholder for the size of the dataset.
num_columns = 28*28
x_train_reshaped = x_train.reshape(-1, num_columns)
x_test_reshaped = x_test.reshape(-1, num_columns)


y_train_cat, y_train_labels = pandas.factorize(y_train, sort=True)
y_test_cat, y_test_labels = pandas.factorize(y_test, sort=True)

# The test and train are balanced so they should have the same number of classes
num_labels = len(set( list(y_test_labels) + list(y_test_labels)))
print(num_labels)
assert num_labels == 10

# ### Building and training the neural network

# Building the model
num_units_penultimate = 2**6
model = Sequential()
model.add(Dense(2**7, input_shape=(num_columns,)))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(.2))
model.add(Dense(num_units_penultimate))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(.2))
model.add(Dense(num_labels, activation='softmax')) # two or more classes

# Compiling the model
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

history = model.fit(x_train_reshaped, y_train_cat, epochs=10, batch_size=num_units_penultimate)


# ### Making predictions
predictions_vector = model.predict(x_test_reshaped)
predictions = [numpy.argmax(pred) for pred in predictions_vector]

array_correct = predictions == y_test_cat # numpy arrays
index_correct = numpy.where(array_correct == True)[0][0] # Where returns tuple

plt.imshow(x_test[index_correct], cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.show()
print("The label is", y_test_cat[index_correct])
print("The prediction is", predictions[index_correct])


# Sometimes the model makes mistakes too.
index_incorrect = numpy.where(array_correct == False)[0][0] # Where returns tuple
plt.imshow(x_test[index_incorrect], cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.show()
print("The label is", y_test_cat[index_incorrect])
print("The prediction is", predictions[index_incorrect])


# ### Finding the accuracy of the model on the test set

num_correct = array_correct.sum()
print("The model is correct", num_correct, "times out of", len(y_test_cat))
print("The accuracy is", num_correct/len(y_test_cat))














