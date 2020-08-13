# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:06:01 2018

@author: julio47
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC;
import seaborn as sns;
from sklearn.metrics import accuracy_score;
from sklearn.model_selection import train_test_split;

np.random.seed(5)

x = np.random.randn(200,2)

y = np.concatenate([[-1]*100, [1]*100])

x[y==1] = x[y==1] + 3.275

y = y.reshape(200, 1)

dat = np.hstack([x,y])

dataset = pd.DataFrame(data = dat, index = np.arange(200),  columns = ['x1', 'x2', 'y'])

plt.scatter(dataset["x1"], dataset["x2"], c = y)

# Seems good enough

# training errors

cost = [0.001, 0.005, 0.01, 0.05, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 500, 1000]

errors = np.zeros(len(cost))

for i in range(18):
    svc = SVC(C = cost[i], kernel = 'linear')
    svc.fit(x, y.ravel())
    preds = svc.predict(x)
    errors[i] = accuracy_score(y.ravel(), preds)
    

errors * 200

# only one observation is missclassified till cost of 100. 500 and 1000 gives 100% accuracy