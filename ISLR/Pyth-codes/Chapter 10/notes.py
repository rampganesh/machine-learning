# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:36:33 2018

@author: mzayauser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


dataset = pd.read_csv('USArrests.csv', index_col = 0)

dataset.apply(np.mean, 0)

dataset.apply(np.var, 0)

prcomp = PCA(n_components = 4)

scaled = pd.DataFrame(scale(dataset), index = dataset.index, columns = dataset.columns)
 
prout = prcomp.fit(scaled)

# PC loadings
prout.components_

# Variance of the loadings
prout.explained_variance_

# SD of the loadings
np.sqrt(prout.explained_variance_)

# Percentage of variance explained
prout.explained_variance_ratio_

# Principal component of the dataset
pc_data = pd.DataFrame(prout.transform(scaled), index = scaled.index, columns = ['Murder', 'Assault', 'UrbanPop', 'Rape'])


# Taken from 'https://github.com/teddyroland/python-biplot/blob/master/biplot.py'
# Taken from 'https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%2010.ipynb'

# biplot

xvector = prout.components_[0] 
yvector = -prout.components_[1]

xs = pc_data.iloc[:,0] 
ys = -pc_data.iloc[:,1]

## visualize projections

fig, ax1 = plt.subplots(figsize=(10,10))

ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)

ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

for i in range(len(xs)):
    
    ax1.plot(xs[i], ys[i])
    ax1.text(xs[i], ys[i], list(dataset.index)[i], color='b')
    
#    plt.annotate(i, (xs[i], ys[i]), ha = "center")
    
ax2 = ax1.twinx().twiny() 

ax2.set_ylim(-1,1)
ax2.set_xlim(-1,1)

for i in range(len(xvector)):
    
    ax2.arrow(0, 0, xvector[i], yvector[i],
              color='r')
    ax2.text(xvector[i], yvector[i],
             list(pc_data.columns.values)[i], color='r');

#fig.show()
    
# Scree plot
    
plt.figure(figsize=(10,10))

plt.xlabel('Principal Component')

plt.ylabel('Proportion of Variance Explained')

plt.xticks([1,2,3,4])

plt.plot([1,2,3,4], prout.explained_variance_ratio_, '-bo', label = 'Individual')

plt.plot([1,2,3,4], np.cumsum(prout.explained_variance_ratio_), '-rs', label = 'Cumulative')

plt.legend(loc = 2);












########################     NCI60 data lab    #########################################

