# -*- coding: utf-8 -*-
"""
Using logistic regression to predict fracture from 
"""

import Coding_logistic_regression

import pandas
import statsmodels.api as sm

def process_bmd(dataframe):
    """
    Converts string columns to categorical
    Adds BMI column from weight/ (height/100)^2
    """
    
    columns = [
        'sex', 
        'fracture', 
        'medication'
        ]
    column_weight = 'weight_kg'
    column_height = 'height_cm'
    
    dataframe_p = dataframe.copy()
    
    dict_labels = {}
    for column in columns:
        series, labels = pandas.factorize(dataframe[column])
        dataframe_p.loc[:,column] = series
        dict_labels[column] = labels
        dataframe_p.loc[:,column] = dataframe_p[column].astype('category')

    dataframe_p.loc[:, 'bmi'] = dataframe[column_weight]/ ( dataframe[column_height]/100)**2

    return dataframe_p, dict_labels

data = pandas.read_csv("https://www.dropbox.com/s/7wjsfdaf0wt2kg2/bmd.csv?dl=1")
data, dict_labels = process_bmd(data)

columns_predictors = ['age', 'sex', 'bmi', 'bmd']
column_response = 'fracture' 

exog = sm.add_constant(data[columns_predictors])
model = sm.Logit(
    endog = data[column_response],
    exog = exog)

# statsmodels will notify us if separation is perfect
model.raise_on_perfect_prediction = False
model_results = model.fit()
summary = model_results.summary().as_text()

data_array = data[columns_predictors].values.astype('float')
data_normal = (data_array - data_array.mean())/data_array.std()

weights = Coding_logistic_regression.logistic_regression_algorithm(
    data_normal, 
    data[column_response].values,
    learning_rate = 0.01,
    num_epochs = 1000)