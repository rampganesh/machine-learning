# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:00:40 2018

@author: julio47
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from glmnet import ElasticNet;
from sklearn.linear_model import LinearRegression;
from sklearn.preprocessing import scale;
from sklearn.model_selection import train_test_split, KFold, cross_val_score;
from sklearn.metrics import mean_squared_error;
from sklearn.decomposition import PCA;
from sklearn.cross_decomposition import PLSRegression;

dataset = pd.read_csv('College.csv', index_col = 0)

dataset[["Private"]] = pd.get_dummies(dataset[["Private"]]).iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Apps', axis = 1), dataset[["Apps"]], random_state = 1, train_size = 0.5)


############ Linear Model #####################

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

lm_err = mean_squared_error(y_pred, y_test)


########### Ridge Regression ##################

grid = 10**(np.linspace(10, -2, 100))


ridge_reg = ElasticNet(alpha=0, scoring = "mean_squared_error", lambda_path=grid)

ridge_reg.fit(X_train, y_train)

ridge_reg.lambda_max_

ridge_reg.lambda_best_

plt.figure(figsize=(10,7))
plt.errorbar(np.log(ridge_reg.lambda_path_), -ridge_reg.cv_mean_score_, color='r', linestyle='None', marker='o',
             markersize=5, yerr=ridge_reg.cv_standard_error_, ecolor='lightgrey', capsize=4)
             
for ref, txt in zip([ridge_reg.lambda_best_, ridge_reg.lambda_max_], ['Lambda best', 'Lambda max']):
    plt.axvline(x=np.log(ref), linestyle='dashed', color='lightgrey')
    plt.text(np.log(ref), .95*plt.gca().get_ylim()[1], txt, ha='center')
             
plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error');


y_pred = ridge_reg.predict(X_test, lamb=ridge_reg.lambda_max_)

ridge_err = mean_squared_error(y_pred, y_test)


################# Lasso Regression #####################

lasso_reg = ElasticNet(alpha = 1, scoring="mean_squared_error", lambda_path=grid)

lasso_reg.fit(X_train, y_train)

lasso_reg.lambda_best_

lasso_reg.lambda_max_

plt.figure(figsize=(10,7))
plt.errorbar(np.log(lasso_reg.lambda_path_), -lasso_reg.cv_mean_score_, color='r', linestyle='None', marker='o',
             markersize=5, yerr=lasso_reg.cv_standard_error_, ecolor='lightgrey', capsize=4)
             
for ref, txt in zip([lasso_reg.lambda_best_, lasso_reg.lambda_max_], ['Lambda best', 'Lambda max']):
    plt.axvline(x=np.log(ref), linestyle='dashed', color='lightgrey')
    plt.text(np.log(ref), .95*plt.gca().get_ylim()[1], txt, ha='center')
             
plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error');

y_pred = lasso_reg.predict(X_test, lamb=lasso_reg.lambda_max_)

lasso_err = mean_squared_error(y_pred, y_test)

sum(lasso_reg.coef_ == 0)

###################### PCR ###############################

kfold_cv = KFold(n_splits = 10, shuffle = False, random_state = 1)

pca = PCA()

regressor1 = LinearRegression()

X_train_trans = pca.fit_transform(scale(X_train))

n = len(X_train_trans)

mse = []

# Calculate MSE with only the intercept 

score = -1*cross_val_score(regressor1, np.ones((n,1)), y_train, cv=kfold_cv, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 17 principle components, one component at the time.

for i in np.arange(1, 18):
    score = -1*cross_val_score(regressor1, X_train_trans[:,:i], y_train, cv=kfold_cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
    
plt.plot(mse, '^y-'); plt.plot(mse.index(min(mse)), min(mse), 'r^')
plt.xlabel('Number of principal components')
plt.ylabel('MSE')
plt.title('Apps')
plt.xlim(xmin=-1);

# MSE is lowest at 16 components which is one less than the number of predictors

np.cumsum(np.round(pca.explained_variance_ratio_,2))

# 95% of the variance is explained by the 11th PC

pca1 = PCA()

regressor1 = LinearRegression()

X_test_trans = pca1.fit_transform(scale(X_test))[:,:11]

regressor1.fit(X_train_trans[:,:11], y_train)

y_pred = regressor1.predict(X_test_trans)

pcr_err = mean_squared_error(y_pred, y_test)


##################### Partial Least squares ################


kfold_cv = KFold(n_splits = 10, shuffle = False, random_state = 1)

mse = []

for i in np.arange(1, 18):
    pls = PLSRegression(n_components=i);
    score = -1*cross_val_score(pls, scale(X_train), y_train, cv=kfold_cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
plt.plot(mse, '^y-'); plt.plot(mse.index(min(mse)), min(mse), 'r^')
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('Apps')
plt.xlim(xmin=-1); 


pls = PLSRegression(n_components=8)

pls.fit(scale(X_train), y_train)

y_pred = pls.predict(scale(X_test))

pls_err = mean_squared_error(y_pred, y_test)

errors = np.array([lm_err, ridge_err, lasso_err, pcr_err, pls_err])

order = np.arange(len(errors))+1

elem = [str(x) for x in order]

# Bar plot with rankings

plt.figure(figsize=(10,7))

plt.bar(np.arange(len(errors)), errors, align = "center")

for rnk, txt in zip(np.argsort(errors), elem):
    plt.text(rnk, 1.05*errors[rnk], txt, ha='center')

plt.xticks(np.arange(len(errors)), ['Linear', 'Ridge', 'Lasso', 'PCR', 'PLS'], fontsize=10)
plt.xlabel('Models')
plt.ylabel('MSE')
