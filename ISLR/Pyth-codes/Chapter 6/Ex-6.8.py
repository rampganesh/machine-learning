# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:18:49 2018

@author: julio47
"""


import pandas as pd;
import numpy as np;
#from statsmodels.formula import api as sm;
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from glmnet import ElasticNet
from sklearn.linear_model import Ridge, LinearRegression;
from sklearn.preprocessing import scale;
from sklearn.model_selection import train_test_split, KFold, cross_val_score;
from sklearn.metrics import mean_squared_error;
from sklearn.decomposition import PCA;

X = abs(np.random.normal(loc = 10, scale = 5,size = 100))

eps = np.random.normal(loc = 1, scale = .5,size = 100)

Y = 7 + 9*X + 11*X**2 - 2*X**3 + eps



def processSubset(feature_set):
# Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def getBest(k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model




models = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()

for i in range(1,8):
    models.loc[i] = getBest(i)
    
models.loc[2, "model"].summary()

getBest(19)["model"].summary()

models.apply(lambda row: row[1].rsquared, axis=1)

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})
# Set up a 2x2 grid so we can look at 4 plots at once
plt.subplot(2, 2, 1)
# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector
plt.plot(models["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')
# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector
rsquared = models.apply(lambda row: row[1].rsquared, axis=1)
plt.subplot(2, 2, 2)
plt.plot(rsquared)
plt.plot(rsquared.argmax(), rsquared.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')
