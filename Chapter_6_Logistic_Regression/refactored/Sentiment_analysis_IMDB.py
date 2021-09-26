#!/usr/bin/env python
# coding: utf-8

# # Classifying IMDB movie reviews


import turicreate as tc
import utils


movies = tc.SFrame('./IMDB_Dataset.csv')
movies


movies['words'] = tc.text_analytics.count_words(movies['review'])
movies

model = tc.logistic_classifier.create(movies, features=['words'], target='sentiment')

model





weights = model.coefficients
weights





weights.sort('value')





weights.sort('value', ascending=False)





weights[weights['index']=='wonderful']





weights[weights['index']=='horrible']





weights[weights['index']=='the']


# In[ ]:








movies['predictions'] = model.predict(movies, output_type='probability')





movies.sort('predictions', ascending=False)[0]





movies.sort('predictions', ascending=True)[0]


# In[ ]:




