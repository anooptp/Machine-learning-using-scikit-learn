# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:56:01 2016

@author: anooptp
"""


from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import KFold

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)
y_pred =knn.predict(X_test)

print ("train_test_split method: ", metrics.accuracy_score(y_pred, y_test))

#from sklearn.cross_validation import KFold
#kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set
#print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
#for iteration, data in enumerate(kf, start=1):
#    print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))

# Cross-validation example: parameter tuning
from sklearn.cross_validation import cross_val_score

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier()
scores = cross_val_score(knn, X, y,cv =10, scoring='accuracy')

print ("cross_val_score method: ",scores)
print ("cross_val_score method: ",scores.mean())

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
    k_scores.append(scores.mean())
    
print ("k_scores: ", k_scores)

import matplotlib.pyplot as plt

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# Cross-validation example: model selection
# Goal: Compare the best KNN model with logistic regression on the iris dataset
# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors = 20)
print("KNeighborsClassifier(n_neighbors = 20): ", cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print("LogisticRegression: ", cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

# Cross-validation example: feature selection
# Goal: Select whether the Newspaper feature should be included in the linear regression model on the advertising dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read in the advertising dataset
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

print("------- Advertising data -------")
print(data.head())

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# 10-fold cross-validation with all three features
linreg = LinearRegression()
scores = cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')
print ("LinearRegression: ", scores)

# fix the sign of MSE scores
mse_scores = -scores
print("LinearRegression (mse): ", mse_scores)

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print("LinearRegression (rmse): ", rmse_scores)

# calculate the average RMSE
print("LinearRegression (rmse): ", rmse_scores.mean())

# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
print("LinearRegression (excluding Newspaper): ", np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')).mean())

