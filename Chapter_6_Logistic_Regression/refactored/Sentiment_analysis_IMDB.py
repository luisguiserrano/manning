#!/usr/bin/env python
# coding: utf-8

"""
Classifying IMDB movie reviews

Document = text
text => list of words
"""

import Coding_logistic_regression

import matplotlib.pyplot as plt
import numpy
import pandas
import pathlib
import seaborn
import tqdm
# import turicreate as tc
import statsmodels.api as sm

def process_movies(dataframe):

    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tqdm.tqdm.pandas()
    dataframe["words"] = (
        dataframe["review"]
        .str.lower()
        .progress_apply( lambda x: ''.join(filter(whitelist.__contains__, x)))
        .str.split()
        )

    column_label = 'sentiment'
    series, labels = pandas.factorize(dataframe[column_label])

    dataframe.loc[:, 'sentiment_encoded'] = series
    dataframe.loc[:, 'sentiment_encoded'] = dataframe['sentiment_encoded'].astype('category')

    return dataframe, labels

def construct_frequency_dict_from_series(series_words):
    """
    Words are separate then converted to dictionary of word counts
    Obviously words should be cleaned before-hand
    """
    series_counts = (
        series_words
        .explode() # produces row for each item in list
        .value_counts()
    )

    return series_counts.to_dict()

def construct_model_dictionary(dataframe, column_words, column_labels):
    dict_model = {}

    g = dataframe.groupby(column_labels)
    for group, group_df in g:
        dict_frequency = construct_frequency_dict_from_series(group_df[column_words])
        dict_model[group] = dict_frequency

    return dict_model

def create_feature(list_words, dict_model):
    """
    dict_model = { label : { word : frequency } }

    Each feature is a 1 x num_labels vector,
    in which each entry is the total frequency count for all
    given words in each label based on all documents.

    e.g. "do not be sad, be happy" for labels [positive, negative]
    may yield [ 20, 42 ]

    Order of labels doesn't matter,
    as long as they are consistent
    """
    array_feature = numpy.zeros(shape = len(dict_model.keys()))

    for index, label in enumerate(list(dict_model.keys())):
        for word in list_words:
            array_feature[index] += dict_model[label].get(word, 0)

    return array_feature

def create_dataframe_features(iter_list_words, dict_model):
    list_features = []
    for list_words in tqdm.tqdm(iter_list_words):
        array_feature = create_feature(list_words, dict_model)
        list_features.append(array_feature)
    return pandas.DataFrame(data = list_features, columns = dict_model.keys())

def check_heatmap(dataframe):
    corr = dataframe.corr()
    kot = corr[corr>=.9]
    plt.figure(figsize=(18,10))
    seaborn.heatmap(kot, cmap="Greens")
    plt.show()

def predict_sentiment(string, dict_model, model):
    """
    The probability of a tweet being positive or negative
    """

    # extract the features of the tweet and store it into x
    array_feature = create_feature(
        string.lower().split(),
        dict_model
        )
    array_feature = numpy.insert(array_feature, 0, 1)

    return model.predict(array_feature)

if __name__ == "__main__":
    dir_data = pathlib.Path('./IMDB_Dataset.csv')
    data = pandas.read_csv(dir_data)
    
    column_text = 'review'
    column_labels = 'sentiment'
    column_labels_encoded = 'sentiment_encoded'
    column_words = "words"
    
    tqdm.tqdm.pandas()
    data, labels = process_movies(data)
    dict_model = construct_model_dictionary(data, column_words, column_labels)
    dataframe_features = create_dataframe_features(data[column_words], dict_model)
    
    exog = sm.add_constant(dataframe_features) # adds an intercept column
    model_logistic_regression = sm.Logit(
        endog = data['sentiment_encoded'],
        exog = exog)
    
    # statsmodels will notify us if separation is perfect
    model_logistic_regression.raise_on_perfect_prediction = False
    results_regression = model_logistic_regression.fit()
    
    ## Examples
    
    list_examples = [
        'I am happy',
        'I am bad',
        'this movie should have been great.',
        'great',
        'great ' * 2,
        'great ' * 3,
        'great ' * 4,
        'I am learning :)',
        'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!',
        'terrible dreadful awful',
        ]
    
    for text in list_examples:
        prediction = predict_sentiment(text, dict_model, results_regression)[0]
        if prediction > 0.5:
            print("{} was {}% negative".format(text, round(prediction*100,2)))
        else:
            print("{} was {}% positive".format(text, 100-round(prediction*100,2)))


    # model = tc.logistic_classifier.create(dataframe_movies, features=['words'], target='sentiment')
    
    # weights = model.coefficients
    
    # weights.sort('value', ascending=False)
    
    # weights[weights['index']=='wonderful']
    
    # weights[weights['index']=='horrible']
    
    # weights[weights['index']=='the']
    
    # dataframe_movies['predictions'] = model.predict(dataframe_movies, output_type='probability')
    
    # dataframe_movies.sort('predictions', ascending=False)[0]
    
    # dataframe_movies.sort('predictions', ascending=True)[0]
