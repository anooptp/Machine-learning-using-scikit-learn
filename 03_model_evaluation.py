# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:35:27 2016

@author: anooptp
"""

# Evaluation procedure #1: Train and test on the entire dataset

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

# create X (features) and y (response)
X= iris.data
y= iris.target

# Logistic regression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
y_pred = logreg.predict(X)

#print(y_pred)

# compute classification accuracy for the logistic regression model
print("------Train and test on the entire dataset------")
print("LogisticRegression Accuracy: ",metrics.accuracy_score(y, y_pred))

# KNN (K=5)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)
y_pred = knn.predict(X)

print("KNeighborsClassifier(K=5) Accuracy: ",metrics.accuracy_score(y, y_pred))

# KNN (K=1)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X, y)
y_pred = knn.predict(X)

print("KNeighborsClassifier(K=1) Accuracy: ",metrics.accuracy_score(y, y_pred))

# Evaluation procedure #2: Train/test split
print("\n------Train/test split------")
# from sklearn.cross_validation import train_test_split
# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

print(X_train.shape,", ", X_test.shape)
print(y_train.shape,", ", y_test.shape)

# STEP 2: train the model on the training set
# LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)
print("LogisticRegression Accuracy: ",metrics.accuracy_score(y_test, y_pred))

# KNN(K=5)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNeighborsClassifier(K=5) Accuracy: ",metrics.accuracy_score(y_test, y_pred))

# KNN(K=1)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNeighborsClassifier(K=1) Accuracy: ",metrics.accuracy_score(y_test, y_pred))

# Can we locate an even better value for K?
# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
print(scores)

# import Matplotlib (scientific plotting library)
# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)

# make a prediction for an out-of-sample observation
print("KNeighborsClassifier(K=11): Prediction:", knn.predict([[3, 5, 4, 2]]))
