# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:15:38 2018

@author: julio47
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC;
import seaborn as sns;
from sklearn.metrics import accuracy_score;
from sklearn.linear_model import LogisticRegression;

np.random.seed(1)

x1 = np.random.randn(500) - 0.5

x2 = np.random.randn(500) - 0.5

y = 1 * ((np.power(x1,2)-np.power(x2,2)) > 0)


# plot the points with class labels

plt.scatter(x1, x2, c = y)

X = pd.DataFrame(data = np.hstack([x1.reshape(500,1),x2.reshape(500,1),y.reshape(500,1)]), 
                 index = np.arange(500), columns = ['x1','x2','y'])


# Logistic regression

log_reg = LogisticRegression(solver = 'newton-cg')

log_reg.fit(X, y)

preds = log_reg.predict(X)

accuracy_score(y, preds)

# 73% accuracy

X = pd.concat([X, pd.DataFrame({'preds' : preds})], axis = 1)

sns.lmplot(data = X, x = 'x1', y = 'x2', hue = 'preds', fit_reg=False)

# The demarcation is clearly linear


# Trying interactions between the predictors and transformations of predictors


# x1*x2


# Trying kernels

svc_linear = SVC(C = 10, kernel = 'linear')

svc_linear.fit(X[["x1","x2"]], y)

preds = svc_linear.predict(X[["x1",'x2']])

accuracy_score(y, preds)

# 72.5% accuracy

# PLot results

plt.scatter(x1, x2, c = preds)



# radial kernel

svc_radial = SVC(C = 10, kernel = 'rbf', gamma = 2)

svc_radial.fit(X[["x1","x2"]], y)

preds = svc_radial.predict(X[["x1",'x2']])

accuracy_score(y, preds)

# 98.5% accuracy

# PLot results

plt.scatter(x1, x2, c = preds)

# The decision boundary is clearly non linear


# Polynomial kernel

svc_poly = SVC(C = 10, kernel = 'poly', degree = 2)

svc_poly.fit(X[["x1","x2"]], y)

preds = svc_poly.predict(X[["x1",'x2']])

accuracy_score(y, preds)

# 98.3% accuracy

# PLot results

plt.scatter(x1, x2, c = preds)

# Same for this one. 

# Radial kernel is the best of the lot