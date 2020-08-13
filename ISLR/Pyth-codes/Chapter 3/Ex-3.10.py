# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:17:29 2018

@author: julio47
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

dataset = pd.read_csv('Carseats.csv')

X = dataset.iloc[:, [0,5,9,10]]

# Converting the qualitative vars

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X.iloc[:, 3] = labelencoder.fit_transform(X.iloc[:, 3])
X.iloc[:, 2] = labelencoder.transform(X.iloc[:, 2])

# Fitting model

regressor = smf.ols(formula = "Sales ~ Price+US+Urban", data = X).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.239
Model:                            OLS   Adj. R-squared:                  0.234
Method:                 Least Squares   F-statistic:                     41.52
Date:                Wed, 14 Feb 2018   Prob (F-statistic):           2.39e-23
Time:                        21:41:02   Log-Likelihood:                -927.66
No. Observations:                 400   AIC:                             1863.
Df Residuals:                     396   BIC:                             1879.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     13.0435      0.651     20.036      0.000        11.764    14.323
Price         -0.0545      0.005    -10.389      0.000        -0.065    -0.044
US             1.2006      0.259      4.635      0.000         0.691     1.710
Urban         -0.0219      0.272     -0.081      0.936        -0.556     0.512
==============================================================================
Omnibus:                        0.676   Durbin-Watson:                   1.912
Prob(Omnibus):                  0.713   Jarque-Bera (JB):                0.758
Skew:                           0.093   Prob(JB):                        0.684
Kurtosis:                       2.897   Cond. No.                         628.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Removing the Urban var

regressor = smf.ols(formula = "Sales ~ Price+US", data = X).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.239
Model:                            OLS   Adj. R-squared:                  0.235
Method:                 Least Squares   F-statistic:                     62.43
Date:                Wed, 14 Feb 2018   Prob (F-statistic):           2.66e-24
Time:                        21:52:58   Log-Likelihood:                -927.66
No. Observations:                 400   AIC:                             1861.
Df Residuals:                     397   BIC:                             1873.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     13.0308      0.631     20.652      0.000        11.790    14.271
Price         -0.0545      0.005    -10.416      0.000        -0.065    -0.044
US             1.1996      0.258      4.641      0.000         0.692     1.708
==============================================================================
Omnibus:                        0.666   Durbin-Watson:                   1.912
Prob(Omnibus):                  0.717   Jarque-Bera (JB):                0.749
Skew:                           0.092   Prob(JB):                        0.688
Kurtosis:                       2.895   Cond. No.                         607.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Improved adjusted R-squared


# Outliers and high leverage

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

# Obs 42 appears to be high leverage point


## outliers
outlierplot = plt.plot(results['fitted'], results['std_resids'],  'o')
plt.xlabel('Fitted values')
plt.ylabel('Square Root of |standardized residuals|')
plt.show(outlierplot)

# No evidence of outliers as all observations between -3 and 3