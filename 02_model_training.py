# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:13:22 2016

@author: anooptp
"""

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X=iris.data

# store response vector in "y"
y= iris.target

# print the shapes of X and y
print(X.shape)
print(y.shape)

# scikit-learn 4-step modeling pattern
# Step 1: Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

# Step 2: "Instantiate" the "estimator"
knn = KNeighborsClassifier(n_neighbors = 1)
print(knn)

# Step 3: Fit the model with data (aka "model training")
knn.fit(X, y)

# Step 4: Predict the response for a new observation
print(knn.predict([[3, 5, 4, 2]]))

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))

# Using a different value for K
# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
print(knn.predict(X_new))

# Using a different classification model
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
print(logreg.predict(X_new))

