# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:18:23 2018

@author: julio47
"""

import matplotlib as plt;
import numpy as np;
import pandas as pd;
from sklearn.svm import SVC;
from sklearn.metrics import accuracy_score;
from sklearn.model_selection import GridSearchCV;

dataset = pd.read_csv('Auto.csv')

dataset.head()

dataset["y"] = np.where(dataset['mpg'] < np.median(dataset['mpg']), 0, 1)

# SVC - cross validation using grid search

cost = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6,7, 8, 9, 10, 50, 100, 500, 1000]

param_grid = dict({'C' : cost})

svc_grid = GridSearchCV(SVC(kernel = 'linear'), param_grid = param_grid, scoring = 'accuracy', cv = 10)

svc_grid.fit(dataset.drop(['mpg','y','name'], axis = 1), dataset.y)


# SVM - radial kernel

gamma = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]

param_grid = dict({'C' : cost, 'gamma' : gamma})

radial_grid = GridSearchCV(SVC(kernel = 'rbf'), param_grid = param_grid, scoring = 'accuracy', cv = 10)

radial_grid.fit(dataset.drop(['mpg','y','name'], axis = 1), dataset.y)


# SVM - poly kernel

degree = [0.1, 0.5, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]

param_grid = dict({'C' : cost, 'degree' : degree})

poly_grid = GridSearchCV(SVC(kernel = 'poly'), param_grid = param_grid, scoring = 'accuracy', cv = 10)

poly_grid.fit(dataset.drop(['mpg','y','name'], axis = 1), dataset.y)