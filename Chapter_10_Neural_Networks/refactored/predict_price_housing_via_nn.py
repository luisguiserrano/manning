
# # Using a regression neural network to predict data_housing prices in Hyderabad

import numpy
import pandas
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU

# Setting random seeds to get reproducible results
numpy.random.seed(0)
tensorflow.random.set_seed(1)

def process_housing(dataframe):
    """
    The columns appear to be categorical, so we won't scale
    The 9's appear to be NAN, so we'll drop those.
    """
    # Find any row with a 9
    mask_has_value = dataframe.iloc[:, :] == 9
    mask_has_value = mask_has_value.any(axis=1)

    return dataframe[~mask_has_value]

def standardize(matrix):
    """
    Adds a tiny offset to prevent dividing by 0
    """
    offset = 1e-6
    mean = numpy.mean(matrix)
    std = numpy.std(matrix) + offset
    numerator = numpy.subtract(matrix,mean)

    return numpy.divide(numerator, std)

def process_features(dataframe):
    """
    Removes Location since it's text
    Standardizes non-categorical columns
    """
    columns_drop = ['Price', 'Location']
    columns_continuous = ['Area'] # 'No. of Bedrooms'

    dataframe_p = (
        dataframe.drop(columns=columns_drop)
        .pipe(standardize)
    )

    return dataframe_p

# ### Loading and preprocessing the dataset
data_housing_raw = pandas.read_csv('Hyderabad.csv')
data_housing = process_housing(data_housing_raw)

column_label = "Price"
features = process_features(data_housing)
labels = data_housing[column_label].values

# ### Building and training the neural network

# Check if the model can actually learn the data
model = Sequential()
model.add(Dense(features.shape[1], input_shape=(features.shape[1],)))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(.2))
model.add(Dense(2**8))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(.2))
model.add(Dense(2**7))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(.2))
model.add(Dense(1)) # predicting number

# Building the model


# Compiling the model. The metrics flag is added for the model to report the root mean squared error at each epoch.
model.compile(
    loss = 'mean_squared_error',
    optimizer='adam',
    metrics=[tensorflow.keras.metrics.RootMeanSquaredError()]
)
model.summary()

history = model.fit(features, labels, epochs=300, batch_size=2**7, verbose = 1)
min(history.history['loss'])
min(history.history['root_mean_squared_error'])

# ### Evaluating the model and making predictions
model.evaluate(features, labels)
model.predict(features)

labels





