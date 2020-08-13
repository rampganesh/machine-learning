# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:23:06 2018

@author: mzayauser
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.formula import api as smf
from statsmodels import api as sm
#from statsmodels.sandbox.regression.predstd import 
from statsmodels.regression import linear_model
#from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import os
import math
import matplotlib


dataset = pd.read_csv("Auto.csv")

sns.set(style="ticks")

# Pairplot

sns.pairplot(dataset)

# Correlation matrix

dataset.iloc[:,0:8].corr()

#                     mpg  cylinders displacement horsepower     weight acceleration       year     origin
# mpg           1.0000000 -0.7776175   -0.8051269 -0.7784268 -0.8322442    0.4233285  0.5805410  0.5652088
# cylinders    -0.7776175  1.0000000    0.9508233  0.8429834  0.8975273   -0.5046834 -0.3456474 -0.5689316
# displacement -0.8051269  0.9508233    1.0000000  0.8972570  0.9329944   -0.5438005 -0.3698552 -0.6145351
# horsepower   -0.7784268  0.8429834    0.8972570  1.0000000  0.8645377   -0.6891955 -0.4163615 -0.4551715
# weight       -0.8322442  0.8975273    0.9329944  0.8645377  1.0000000   -0.4168392 -0.3091199 -0.5850054
# acceleration  0.4233285 -0.5046834   -0.5438005 -0.6891955 -0.4168392    1.0000000  0.2903161  0.2127458
# year          0.5805410 -0.3456474   -0.3698552 -0.4163615 -0.3091199    0.2903161  1.0000000  0.1815277
# origin        0.5652088 -0.5689316   -0.6145351 -0.4551715 -0.5850054    0.2127458  0.1815277  1.0000000

# Possible collinearity between predictors cylinders, displacement, horsepower,weight

# Regression fit

X = dataset.iloc[:,1:8]

y = dataset.iloc[:,0]

regressor = smf.ols(formula = "mpg ~ cylinders+displacement+horsepower+weight+acceleration+year+origin", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.821
Model:                            OLS   Adj. R-squared:                  0.818
Method:                 Least Squares   F-statistic:                     252.4
Date:                Mon, 12 Feb 2018   Prob (F-statistic):          2.04e-139
Time:                        18:11:42   Log-Likelihood:                -1023.5
No. Observations:                 392   AIC:                             2063.
Df Residuals:                     384   BIC:                             2095.
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [95.0% Conf. Int.]
--------------------------------------------------------------------------------
Intercept      -17.2184      4.644     -3.707      0.000       -26.350    -8.087
cylinders       -0.4934      0.323     -1.526      0.128        -1.129     0.142
displacement     0.0199      0.008      2.647      0.008         0.005     0.035
horsepower      -0.0170      0.014     -1.230      0.220        -0.044     0.010
weight          -0.0065      0.001     -9.929      0.000        -0.008    -0.005
acceleration     0.0806      0.099      0.815      0.415        -0.114     0.275
year             0.7508      0.051     14.729      0.000         0.651     0.851
origin           1.4261      0.278      5.127      0.000         0.879     1.973
==============================================================================
Omnibus:                       31.906   Durbin-Watson:                   1.309
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               53.100
Skew:                           0.529   Prob(JB):                     2.95e-12
Kurtosis:                       4.460   Cond. No.                     8.59e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 8.59e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Diagnostic plots


results = pd.DataFrame({'resids': regressor.resid,
                        'std_resids': regressor.resid_pearson,
                        'fitted': regressor.predict()})

print(results.head())

## raw residuals vs. fitted
residsvfitted = plt.plot(results['fitted'], results['resids'],  'o')
l = plt.axhline(y = 0, color = 'grey', linestyle = 'dashed')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show(residsvfitted)

## q-q plot
qqplot = sm.qqplot(results['std_resids'], line='s')
plt.show(qqplot)

## scale-location
scalelocplot = plt.plot(results['fitted'], abs(results['std_resids'])**.5,  'o')
plt.xlabel('Fitted values')
plt.ylabel('Square Root of |standardized residuals|')
plt.title('Scale-Location')
plt.show(scalelocplot)

## residuals vs. leverage
residsvlevplot = sm.graphics.influence_plot(regressor, criterion = 'Cooks', size = 2)
plt.show(residsvlevplot)

# 4 plots in one window
fig = plt.figure(figsize = (8, 8), dpi = 100)

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(results['fitted'], results['resids'],  'o')
l = plt.axhline(y = 0, color = 'grey', linestyle = 'dashed')
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted')

ax2 = fig.add_subplot(2, 2, 2)
sm.qqplot(results['std_resids'], line='s', ax = ax2)
ax2.set_title('Normal Q-Q')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(results['fitted'], abs(results['std_resids'])**.5,  'o')
ax3.set_xlabel('Fitted values')
ax3.set_ylabel('Sqrt(|standardized residuals|)')
ax3.set_title('Scale-Location')

ax4 = fig.add_subplot(2, 2, 4)
sm.graphics.influence_plot(regressor, criterion = 'Cooks', size = 2, ax = ax4)

plt.tight_layout()

# PLaying with the vars

regressor = smf.ols(formula = "mpg ~ displacement+weight+year*origin", data = dataset).fit()

regressor.summary()

regressor = smf.ols(formula = "mpg ~ displacement+year+weight*origin", data = dataset).fit()

regressor.summary()

regressor = smf.ols(formula = "mpg ~ displacement*weight+year+origin", data = dataset).fit()

regressor.summary()


# Polynomial and log effects

#dataset[["weight","year","origin"]] = dataset[["weight","year","origin"]].astype(float)

regressor = smf.ols(formula = "mpg ~ displacement+np.log(weight)+year+origin", data = dataset).fit()

regressor.summary()

regressor = smf.ols(formula = "mpg ~ np.log(displacement)+year+weight+origin", data = dataset).fit()

regressor.summary()

regressor = smf.ols(formula = "mpg ~ displacement+weight+year+np.power(origin,2)", data = dataset).fit()

regressor.summary()

regressor = smf.ols(formula = "mpg ~ displacement+np.sqrt(weight)+year+origin", data = dataset).fit()

regressor.summary()

regressor = smf.ols(formula = "mpg ~ I(np.power(displacement,2))+weight+year+origin", data = dataset).fit()

regressor.summary()

# seems to have the most effect