# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:12:14 2018

@author: julio47

The code is not written with the intention of replicating R results as
given in the book, rather, show how it can be done entirely in Python.

gca()
itertools
errorbar()
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

dataset = pd.read_csv('Hitters.csv', index_col = 0)

# Equivalent to na.omit in R

def isna(row):
    return(all(row.notnull()));
    
dataset = dataset[dataset.apply(isna, axis = 1)]

dummies = pd.get_dummies(dataset[['League', 'Division', 'NewLeague']])

y = dataset.Salary
# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = dataset.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# Taken from 'http://www.science.smith.edu/~jcrouser/SDS293/labs/2016/lab8/Lab%208%20-%20Subset%20Selection%20in%20Python.pdf'

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


############# Forward Selection #########################

models2 = pd.DataFrame(columns=["RSS", "model"])


def forward(predictors):
# Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p]))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

tic = time.time()

predictors = []

for i in range(1,len(X.columns)+1):
    models2.loc[i] = forward(predictors)
    predictors = models2.loc[i]["model"].model.exog_names;
    
    
toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")


print(models.loc[1, "model"].summary())
print(models.loc[2, "model"].summary())


############### Backward Selection #######################


def backward(predictors):
    tic = time.time()
    results = []
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(processSubset(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)-1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model



models3 = pd.DataFrame(columns=["RSS", "model"], index = range(1,len(X.columns)))
tic = time.time()
predictors = X.columns

while(len(predictors) > 1):
    models3.loc[len(predictors)-1] = backward(predictors)
    predictors = models3.loc[len(predictors)-1]["model"].model.exog_names


toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

print(models.loc[7, "model"].params)
print(models2.loc[7, "model"].params)
print(models3.loc[7, "model"].params)



################## Ridge Regression #########################

# Taken from 'https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%206.ipynb'

grid = np.power(10, np.linspace(10, -2, 100))

models = []

coefs = []

regressor = Ridge()

for a in grid:
    regressor.set_params(alpha=a)
    regressor.fit(scale(X), y)
    coefs.append(regressor.coef_)
    models.append({'model' : regressor})
    
#pd.DataFrame(data=coefs, columns = np.append(['lambda'], dataset.columns.values))

ax = plt.gca()
ax.plot(grid, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization');


# predict for lambda = 50

regressor = ElasticNet(alpha = 0, lambda_path=grid)

regressor.fit(scale(X), y)

regressor.lambda_best_

regressor.lambda_path[49]

regressor.lambda_path[55]

regressor.predict(scale(X),lamb=50)

# SPlitting dataset to train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, train_size = 0.5)

regressor.fit(scale(X_train), y_train)

# checking results for lambda 4

results = regressor.predict(scale(X_test), lamb=4)

mean_squared_error(results, y_test)

# Choosing lambda by Cross validation. 
# Note that we have already split the dataset

regressor1 = ElasticNet(alpha = 0, scoring="mean_squared_error", tol=1e-12)

regressor1.fit(scale(X_train), y_train)

regressor1.lambda_max_

regressor1.lambda_best_

results = regressor1.predict(scale(X_test), lamb=regressor1.lambda_max_)

mean_squared_error(results, y_test)

# Plotting the validation errors
# Taken from 'https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%206.ipynb'


plt.figure(figsize=(10,7))
plt.errorbar(np.log(regressor1.lambda_path_), -regressor1.cv_mean_score_, color='r', linestyle='None', marker='o',
             markersize=5, yerr=regressor1.cv_standard_error_, ecolor='lightgrey', capsize=4)
             
for ref, txt in zip([regressor1.lambda_best_, regressor1.lambda_max_], ['Lambda best', 'Lambda max']):
    plt.axvline(x=np.log(ref), linestyle='dashed', color='lightgrey')
    plt.text(np.log(ref), .95*plt.gca().get_ylim()[1], txt, ha='center')
             
plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error');


##################### Lasso Regression ###########################


lasso = ElasticNet(alpha=1, lambda_path=grid, scoring="mean_squared_error")

lasso.fit(scale(X_train), y_train)


l1_norm = np.sum(np.abs(lasso.coef_path_), axis=0)

plt.figure(figsize=(10,6))
plt.plot(l1_norm, lasso.coef_path_.T)
plt.xlabel('L1 norm')
plt.ylabel('Coefficients');

# You can see evidence of feature selection. 

plt.figure(figsize=(10,7))
plt.errorbar(np.log(lasso.lambda_path_), -lasso.cv_mean_score_, color='r', linestyle='None', marker='o',
             markersize=5, yerr=lasso.cv_standard_error_, ecolor='lightgrey', capsize=4)
             
for ref, txt in zip([lasso.lambda_best_, lasso.lambda_max_], ['Lambda best', 'Lambda max']):
    plt.axvline(x=np.log(ref), linestyle='dashed', color='lightgrey')
    plt.text(np.log(ref), .95*plt.gca().get_ylim()[1], txt, ha='center')
             
plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error');



########################## Principal Components Regression ############################

pca = PCA()

regressor = LinearRegression()

results = pca.fit_transform(scale(X))

n = len(results)

print(pca.components_.shape)

pd.DataFrame(pca.components_.T).loc[:4,:5]

np.cumsum(np.round(pca.explained_variance_ratio_,2))

# Only 13 Principal components seem to have any effect at all on the variance explained
# of which the first 7 alone explain 92% of the variance in the dataset



# K fold cross validation to select the optimal number of principal components

kfold_cv = KFold(n_splits = 10, shuffle = True, random_state = 1)

mse = []

# Calculate MSE with only the intercept 

score = -1*cross_val_score(regressor, np.ones((n,1)), y, cv=kfold_cv, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 19 principle components, one component at the time.

for i in np.arange(1, 20):
    score = -1*cross_val_score(regressor, results[:,:i], y, cv=kfold_cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
    
plt.plot(mse, '^y-'); plt.plot(mse.index(min(mse)), min(mse), 'r^')
plt.xlabel('Number of principal components')
plt.ylabel('MSE')
plt.title('Salary')
plt.xlim(xmin=-1);

# MSE is lowest at Principal components 18


# Performing 10 fold cross validation on training dataset

pca1 = PCA()

regressor1 = LinearRegression()

X_train_trans = pca1.fit_transform(scale(X_train))

n = len(X_train_trans)

mse1 = []

# Calculate MSE with only the intercept 

score1 = -1*cross_val_score(regressor1, np.ones((n,1)), y_train, cv=kfold_cv, scoring='neg_mean_squared_error').mean()    
mse1.append(score1)

# Calculate MSE using CV for the 19 principle components, one component at the time.

for i in np.arange(1, 20):
    score1 = -1*cross_val_score(regressor1, X_train_trans[:,:i], y_train, cv=kfold_cv, scoring='neg_mean_squared_error').mean()
    mse1.append(score1)
    
    
plt.plot(mse1, '^y-'); plt.plot(mse1.index(min(mse1)), min(mse1), 'r^')
plt.xlabel('Number of principal components')
plt.ylabel('MSE')
plt.title('Salary')
plt.xlim(xmin=-1);

# The MSE is lowest when performing PCR with only 6 components.

# Fitting the linear model on only 6 components

X_test_trans = pca1.transform(scale(X_test))[:,:6]

regressor = LinearRegression()

regressor.fit(X_train_trans[:,:6], y_train)

y_pred = regressor.predict(X_test_trans)

mean_squared_error(y_test, y_pred)