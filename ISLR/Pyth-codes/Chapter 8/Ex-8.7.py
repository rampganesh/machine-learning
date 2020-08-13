# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:04:48 2018

@author: julio47
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split, GridSearchCV;
from sklearn.metrics import mean_squared_error;
from sklearn.ensemble import RandomForestRegressor;

dataset = pd.read_csv('Boston.csv', index_col = 0)

X = dataset.drop(['medv'], axis = 1)

y = dataset["medv"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.5)

estimators = list(range(25,101))

features = [13, 6, 3]

errmat = np.zeros((3, 76))

for i in range(0,3):
    for j in range(0,76):
        regressor = RandomForestRegressor(n_estimators=estimators[j], max_features=features[i], random_state=1, n_jobs=-1)
        regressor.fit(X_train, y_train)
        preds = regressor.predict(X_test)
        errmat[i,j] = mean_squared_error(y_test, preds)
        print("At ", i, j)

plt.figure(figsize=(10,7))
plt.plot(estimators, errmat[0], label="{:g}".format(features[0]))
plt.plot(estimators, errmat[1], label="{:g}".format(features[1]))
plt.plot(estimators, errmat[2], label="{:g}".format(features[2]))
plt.legend()





#grid = GridSearchCV(regressor, param_grid, cv = 10, scoring = "neg_mean_squared_error")

#grid.fit(X, y)