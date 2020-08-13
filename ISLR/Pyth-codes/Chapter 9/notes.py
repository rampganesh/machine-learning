# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:01:33 2018

@author: julio47

SVM plot code from : 'https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%209.ipynb'
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC;
import seaborn as sns;
from sklearn.model_selection import GridSearchCV, train_test_split;
from sklearn.metrics import confusion_matrix, auc, roc_curve;

np.random.seed(1)

x = np.random.randn(20,2)

y = np.concatenate([[-1]*10, [1]*10])

y = y.reshape(20, 1)

dat = np.hstack([x,y])

dataset = pd.DataFrame(data = dat, index = np.arange(20),  columns = ['x1', 'x2', 'y'])

# Padding one set of values to make them linearly separable

dataset.loc[dataset.y==1, 'x1'] = dataset.loc[dataset.y==1, 'x1']+1.5

dataset.loc[dataset.y==1, 'x2'] = dataset.loc[dataset.y==1, 'x2']+1.5

sns.lmplot(x = 'x1', y = 'x2', data = dataset, fit_reg = False, hue = 'y', legend = False)

svc = SVC(C= 1.0, kernel='linear')

svc.fit(dataset[["x1",'x2']], dataset[['y']])



def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)


plot_svc(svc, dataset[["x1",'x2']].values, dataset[['y']].values)

# Smaller value for the cost parameter

svc = SVC(C= 0.1, kernel='linear')

svc.fit(dataset[["x1",'x2']], dataset[['y']])

plot_svc(svc, dataset[["x1",'x2']].values, dataset[['y']].values)


# Trying a range of values for Cost

cost = [0.001, 0.01, 0.1, 1, 5, 10, 100]

param_grid = dict({'C':cost})

gridsearch = GridSearchCV(SVC(kernel = 'linear'), param_grid, scoring = 'accuracy', cv = 10, return_train_score = True)

gridsearch.fit(dataset[["x1",'x2']].values, dataset[['y']].values.flatten())

gridsearch.best_score_

gridsearch.best_params_

#### Testing the best parameters

np.random.seed(2)

x_test = np.random.randn(20,2)

y_test = np.concatenate([[-1]*10, [1]*10])

y_test = y_test.reshape(20, 1)

dat = np.hstack([x_test,y_test])

dataset1 = pd.DataFrame(data = dat, index = np.arange(20),  columns = ['x1', 'x2', 'y'])

# Padding one set of values to make them linearly separable

dataset1.loc[dataset1.y==1, 'x1'] = dataset1.loc[dataset1.y==1, 'x1']+1.5

dataset1.loc[dataset1.y==1, 'x2'] = dataset1.loc[dataset1.y==1, 'x2']+1.5

# Model initialization

svc = SVC(C = 0.001, kernel = 'linear')

svc.fit(dataset[["x1",'x2']], dataset[['y']])

preds = svc.predict(dataset1[["x1",'x2']])

confusion_matrix(y_test.flatten(), preds)

# 80% accuracy


plt.scatter(dataset.x1, dataset.x2, c = dataset.y, s = 70)

# Test data

plt.plot(dataset1.x1, dataset1.x2, 'go')



# Reducing the distance between the two classes

dat = np.hstack([x_test,y_test])

dataset1 = pd.DataFrame(data = dat, index = np.arange(20),  columns = ['x1', 'x2', 'y'])

# Padding one set of values to make them linearly separable

dataset1.loc[dataset1.y==1, 'x1'] = dataset1.loc[dataset1.y==1, 'x1']+1

dataset1.loc[dataset1.y==1, 'x2'] = dataset1.loc[dataset1.y==1, 'x2']+1

plt.scatter(dataset1.x1, dataset1.x2, c = dataset1.y, s = 70)

# Model initialization

svc = SVC(C = 1e5, kernel = 'linear')

svc.fit(dataset1[["x1",'x2']], dataset1[['y']])

plot_svc(svc, dataset1[["x1",'x2']].values, dataset1[['y']].values)


#################### Support Vector Machines ########################

np.random.seed(3)

X = np.random.randn(200,2)

y = np.concatenate([[-1]*150, [1]*50])

y = y.reshape(200,1)

X[:100] = X[:100] + 2

X[101:150] = X[101:150] - 2

# Visualize the data

plt.scatter(X[:,0], X[:,1], s = 70, c = y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.5)

svm = SVC(C = 1)

svm.fit(X_train, y_train.ravel())

plot_svc(svm, X_train, y_train)

# Increase the cost


svm1 = SVC(C = 1)

svm1.fit(X_train, y_train.ravel())

plot_svc(svm1, X_train, y_train)

# Performing Grid searchCV to find best parameters

param_grid = dict({'C' : [0.1,1,10,100,1000], 'gamma' : [0.5, 1, 2, 3, 4]})

gridsearch = GridSearchCV(SVC(kernel = 'rbf'), param_grid, scoring = 'accuracy', cv = 10, return_train_score=True)

gridsearch.fit(X_train, y_train.ravel())

gridsearch.best_params_

gridsearch.best_score_

# Testing Accuracy using the best params

svc = SVC(C = 1, gamma = 0.5)

svc.fit(X_train, y_train.ravel())

preds = svc.predict(X_test)

confusion_matrix(y_test, preds)

# 82% accuracy rate

################### ROC Curves #####################

svm3 = SVC(C = 1, gamma = 2)

svm3.fit(X_train, y_train.ravel())


# A more flexible fit

svm4 = SVC(C = 1, gamma = 50)

svm4.fit(X_train, y_train.ravel())

y_train_score3 = svm3.decision_function(X_train)
y_train_score4 = svm4.decision_function(X_train)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax1.set_title('Training Data')

y_test_score3 = svm3.decision_function(X_test)
y_test_score4 = svm4.decision_function(X_test)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax2.set_title('Test Data')

for ax in fig.axes:
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

