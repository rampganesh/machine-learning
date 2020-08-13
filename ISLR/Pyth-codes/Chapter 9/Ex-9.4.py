# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:28:37 2018

@author: julio47

SVM plot code from : 'https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%209.ipynb'
"""
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC;
from sklearn.metrics import confusion_matrix;


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


np.random.seed(1)

x = np.random.randn(100,2)

y = np.concatenate([[-1]*50, [1]*50])

x[y==1] = x[y==1] + 2.2

y = y.reshape(100, 1)

dat = np.hstack([x,y])

dataset = pd.DataFrame(data = dat, index = np.arange(100),  columns = ['x1', 'x2', 'y'])

plt.scatter(dataset["x1"], dataset["x2"], c = y)


# Linear kernel

svc = SVC(C = 10.0, kernel = "linear")

svc.fit(dataset[["x1",'x2']], dataset[['y']])

plot_svc(svc, dataset[["x1",'x2']].values, dataset[['y']].values)


preds = svc.predict(dataset[["x1",'x2']])

confusion_matrix(dataset[['y']].values, preds)

# 97% accuracy


# radial

svc_radial = SVC(C = 10.0, kernel = "rbf", gamma = 1)

svc_radial.fit(dataset[["x1",'x2']], dataset[['y']])

plot_svc(svc_radial, dataset[["x1",'x2']].values, dataset[['y']].values)


preds = svc_radial.predict(dataset[["x1",'x2']])

confusion_matrix(dataset[['y']].values, preds)

# 100% accuracy


# polynomial

svc_poly = SVC(C = 10.0, kernel = "poly")

svc_poly.fit(dataset[["x1",'x2']], dataset[['y']])

plot_svc(svc_poly, dataset[["x1",'x2']].values, dataset[['y']].values)


# seems to be the best

preds = svc_poly.predict(dataset[["x1",'x2']])

confusion_matrix(dataset[['y']].values, preds)

# 100 % accuracy

# testing test error


np.random.seed(3)

x = np.random.randn(50,2)

y = np.concatenate([[-1]*25, [1]*25])

x[y==1] = x[y==1] + 2.2

y = y.reshape(50, 1)

dat = np.hstack([x,y])

dataset = pd.DataFrame(data = dat, index = np.arange(50),  columns = ['x1', 'x2', 'y'])

plt.scatter(dataset["x1"], dataset["x2"], c = y)

# polynomial kernel

preds = svc_poly.predict(dataset[["x1",'x2']])

confusion_matrix(dataset[['y']].values, preds)

# 94% accuracy

# radial kernel

preds = svc_radial.predict(dataset[["x1",'x2']])

confusion_matrix(dataset[['y']].values, preds)

# 96% accuracy

