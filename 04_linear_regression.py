# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 00:01:24 2016

@author: anooptp
"""

import pandas as pd

# read CSV file directly from a URL and save the results
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# display the first 5 rows
print (data.head())
print(data.shape)

# Visualizing data using seaborn
import seaborn as sns

# visualize the relationship between the features and the response using scatterplots
#sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=5, aspect=0.7, kind='reg')

# create a Python list of feature names
feature_cols = ['TV', 'Radio', 'Newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
X = data[['TV', 'Radio', 'Newspaper']]
print(X.head())

# check the type and shape of X
print(type(X))
print(X.shape)

y = data['Sales']

print(y.head())

# check the type and shape of y
print(type(y))
print(y.shape)

# Splitting X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Linear regression in scikit-learn
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print(linreg.intercept_)
print (linreg.coef_)

# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# Computing the RMSE for our Sales predictions
from sklearn import metrics
import numpy as np

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Since Newspaper doesn't improve the quality of our predictions
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
y = data.Sales

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
