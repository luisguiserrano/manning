#!/usr/bin/env python
# coding: utf-8

"""
Classifying IMDB movie reviews
"""

import matplotlib.pyplot as plt
import seaborn
import numpy
import pandas
import pathlib
import tqdm
# import turicreate as tc
import statsmodels.api as sm

def remove_from_string(string, string_remove):
    """
    Absolute fastest.
    """
    return string.translate(str.maketrans('','', string_remove))

def construct_frequency_dict_from_series(series_text):
    """ 
    Converts text to lower-case then returns dictionary of word counts 
    Obviously one should pass some other cleansing function
    """
    string_remove_spaces = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    string_remove = '0123456789'
    string_remove += '"'
    string_remove += "'"
    
    series_counts = (
        series_text
        .str.lower()
        .str.translate(str.maketrans(' ', ' ', string_remove_spaces))
        .str.translate(str.maketrans('', '', string_remove))
        .str.split() # string -> list
        .explode() # produces row for each item in list
        .value_counts()
    )
    
    return series_counts.to_dict()

def select_column_values_item(dataframe, column, item):
    mask_item = dataframe[column] == item
    return dataframe[mask_item]

def process_movies(dataframe):
    dataframe["words"] = (
        dataframe["review"]
        .str.lower().replace('"', '')
        .str.split()
        )
    
    column_label = 'sentiment'
    series, labels = pandas.factorize(dataframe[column_label])
    
    dataframe.loc[:, 'sentiment_encoded'] = series
    dataframe.loc[:, 'sentiment_encoded'] = dataframe['sentiment_encoded'].astype('category')

    return dataframe, labels

def construct_model_dictionary(dataframe, column_text, column_labels):
    dict_model = {}

    g = dataframe.groupby(column_labels)
    for group, group_df in g:
        dict_frequency = construct_frequency_dict_from_series(group_df[column_text])
        dict_model[group] = dict_frequency

    return dict_model 

def create_feature(list_words, list_labels, dict_model):
    """
    Each feature is a 1 x num_labels vector, 
    in which each entry is the total frequency count for all 
    given words in each label based on all documents.
    
    e.g. "do not be sad, be happy" for labels [positive, negative]
    may yield [ 20, 42 ]
    """
    array_feature = numpy.zeros(shape = len(list_labels))
    
    for word in list_words:
        for index, label in enumerate(list_labels):
            array_feature[index] += dict_model[label].get(word, 0)
    
    return array_feature

def create_dataframe_features(iter_list_words, list_labels, dict_model):
    list_features = []
    for list_words in tqdm.tqdm(iter_list_words):
        array_feature = create_feature(list_words, list_labels, dict_model)
        list_features.append(array_feature)
    return pandas.DataFrame(data = list_features, columns = list_labels)

def check_heatmap(dataframe):
    corr = dataframe.corr() 
    kot = corr[corr>=.9] 
    plt.figure(figsize=(18,10))
    seaborn.heatmap(kot, cmap="Greens")
    plt.show()

dir_data = pathlib.Path('./IMDB_Dataset.csv')
data = pandas.read_csv(dir_data)

column_text = 'review'
column_labels = 'sentiment'
column_labels_encoded = 'sentiment_encoded'
column_words = "words"

tqdm.tqdm.pandas()
data, labels = process_movies(data)
dict_model = construct_model_dictionary(data, column_text, column_labels)
dataframe_features = create_dataframe_features(data[column_words], labels, dict_model)

exog = sm.add_constant(dataframe_features) # adds an intercept column
model_logistic_regression = sm.Logit(
    endog = data['sentiment_encoded'],
    exog = exog)

# statsmodels will notify us if separation is perfect
model_logistic_regression.raise_on_perfect_prediction = False
results_regression = model_logistic_regression.fit()

def predict_sentiment(string, list_labels, dict_model, model):
    """
    Output: 
        y_pred: the probability of a tweet being positive or negative
    """
    
    # extract the features of the tweet and store it into x
    array_feature = create_feature(
        string.lower().split(),
        list_labels,
        dict_model
        )
    array_feature = numpy.insert(array_feature, 0, 1)
    
    return model.predict(array_feature)

## Examples

s = 'glad' * 100
p = predict_sentiment(s, labels, dict_model, results_regression)
print(p)

# model = tc.logistic_classifier.create(dataframe_movies, features=['words'], target='sentiment')

# weights = model.coefficients

# weights.sort('value', ascending=False)

# weights[weights['index']=='wonderful']

# weights[weights['index']=='horrible']

# weights[weights['index']=='the']

# dataframe_movies['predictions'] = model.predict(dataframe_movies, output_type='probability')

# dataframe_movies.sort('predictions', ascending=False)[0]

# dataframe_movies.sort('predictions', ascending=True)[0]
