# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:25:24 2018

@author: julio47
"""

import pandas as pd;
import numpy as np;
from statsmodels.formula import api as smf;
from statsmodels import api as sm;
#from sklearn.linear_model import LinearRegression;
from sklearn.preprocessing import PolynomialFeatures;
from matplotlib import pyplot as plt;
import seaborn as sns;

dataset = pd.read_csv('Wage.csv')

# Taken from 'https://stackoverflow.com/questions/41317127/python-equivalent-to-r-poly-function'

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

def get_range(vals):
    return min(vals.values), max(vals.values)


regressor = smf.ols(formula = "wage ~ poly(age,4)", data = dataset).fit()

regressor.summary()

rangelims = get_range(dataset[['age']])

grid = np.linspace(rangelims[0], rangelims[1]).astype('int').reshape(-1,1)

X = PolynomialFeatures(4).fit_transform(dataset.age.reshape(-1,1))

regressor = sm.GLS(dataset.wage.reshape(-1,1), X)

res = regressor.fit()

# Creating orthogonal polynomials

X1 = PolynomialFeatures(1).fit_transform(dataset.age.reshape(-1,1))
X2 = PolynomialFeatures(2).fit_transform(dataset.age.reshape(-1,1))
X3 = PolynomialFeatures(3).fit_transform(dataset.age.reshape(-1,1))
X4 = PolynomialFeatures(4).fit_transform(dataset.age.reshape(-1,1))
X5 = PolynomialFeatures(5).fit_transform(dataset.age.reshape(-1,1))

y = (dataset.wage > 250).map({False:0, True:1}).as_matrix()

fit_1 = fit = sm.GLS(dataset.wage, X1).fit()
fit_2 = fit = sm.GLS(dataset.wage, X2).fit()
fit_3 = fit = sm.GLS(dataset.wage, X3).fit()
fit_4 = fit = sm.GLS(dataset.wage, X4).fit()
fit_5 = fit = sm.GLS(dataset.wage, X5).fit()

sm.stats.anova_lm(fit_1, fit_2, fit_3, fit_4, fit_5, typ=1)

# Logistic Regression Model

clf = sm.GLM(y, X4, family = sm.families.Binomial(sm.families.links.logit))
res = clf.fit()

X_test = PolynomialFeatures(4).fit_transform(grid)

pred = res.predict(X_test)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
fig.suptitle('Degree-4 Polynomial', fontsize=14)

# Scatter plot with polynomial regression line
ax1.scatter(dataset.age, dataset.wage, facecolor='None', edgecolor='k', alpha=0.3)
sns.regplot(dataset.age, dataset.wage, order = 4, truncate=True, scatter=False, ax=ax1)
ax1.set_ylim(ymin=0)

# Logistic regression showing Pr(wage>250) for the age range.
ax2.plot(grid, pred, color='b')

# Rug plot showing the distribution of wage>250 in the training data.
# 'True' on the top, 'False' on the bottom.
ax2.scatter(dataset.age, y/5, s=30, c='grey', marker='|', alpha=0.7)

ax2.set_ylim(-0.01,0.21)
ax2.set_xlabel('age')
ax2.set_ylabel('Pr(wage>250|age)');



# Step Function

df_cut, bins = pd.cut(dataset.age, 4, retbins=True, right=True)
df_cut.value_counts(sort=False)

df_steps = pd.concat([dataset.age, df_cut, dataset.wage], keys=['age','age_cuts','wage'], axis=1)
df_steps.head(5)


# Create dummy variables for the age groups
df_steps_dummies = pd.get_dummies(df_steps['age_cuts'])

# Statsmodels requires explicit adding of a constant (intercept)
df_steps_dummies = sm.add_constant(df_steps_dummies)

df_steps_dummies.head(5)


# Using statsmodels because it has a more complete output for coefficients
fit3 = sm.GLM(df_steps.wage, df_steps_dummies.drop(['(17.938, 33.5]'], axis=1)).fit()
print(fit3.summary().tables[1])

# Put the test data in the same bins as the training data.
bin_mapping = np.digitize(grid.ravel(), bins)
bin_mapping

# Get dummies, drop first dummy category, add constant
X_test2 = sm.add_constant(pd.get_dummies(bin_mapping).drop(1, axis=1))
X_test2.head()


pred2 = fit3.predict(X_test2)


