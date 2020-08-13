# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:50:52 2018

@author: mzayauser
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.formula import api as smf
#from statsmodels.sandbox.regression.predstd import 
from statsmodels.regression import linear_model
#from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import math
import matplotlib
import random

random.seed(1)

x = np.random.randn(100)

y = 2* x + np.random.randn(100)

dataset = {'x': x, 'y': y}

dataset = pd.DataFrame(dataset)

regressor = smf.ols(formula = "y~x", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.838
Model:                            OLS   Adj. R-squared:                  0.837
Method:                 Least Squares   F-statistic:                     508.7
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           1.40e-40
Time:                        12:16:58   Log-Likelihood:                -139.92
No. Observations:                 100   AIC:                             283.8
Df Residuals:                      98   BIC:                             289.0
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.0579      0.100     -0.582      0.562      -0.255       0.140
x              2.1597      0.096     22.555      0.000       1.970       2.350
==============================================================================
Omnibus:                        0.131   Durbin-Watson:                   2.127
Prob(Omnibus):                  0.936   Jarque-Bera (JB):                0.312
Skew:                          -0.016   Prob(JB):                        0.856
Kurtosis:                       2.728   Cond. No.                         1.11
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

regressor = smf.ols(formula = "x~y", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      x   R-squared:                       0.838
Model:                            OLS   Adj. R-squared:                  0.837
Method:                 Least Squares   F-statistic:                     508.7
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           1.40e-40
Time:                        12:17:32   Log-Likelihood:                -54.110
No. Observations:                 100   AIC:                             112.2
Df Residuals:                      98   BIC:                             117.4
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.0057      0.042      0.134      0.893      -0.078       0.090
y              0.3882      0.017     22.555      0.000       0.354       0.422
==============================================================================
Omnibus:                        1.439   Durbin-Watson:                   2.121
Prob(Omnibus):                  0.487   Jarque-Bera (JB):                1.375
Skew:                          -0.280   Prob(JB):                        0.503
Kurtosis:                       2.874   Cond. No.                         2.48
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


random.seed(1)

x = np.random.normal(1000, 0.1, 100)

y =  np.random.normal(1000, 0.1, 100)

dataset = {'x': x, 'y': y}

dataset = pd.DataFrame(dataset)

regressor = smf.ols(formula = "y~x", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.040
Model:                            OLS   Adj. R-squared:                  0.030
Method:                 Least Squares   F-statistic:                     4.097
Date:                Fri, 23 Feb 2018   Prob (F-statistic):             0.0457
Time:                        12:31:33   Log-Likelihood:                 94.950
No. Observations:                 100   AIC:                            -185.9
Df Residuals:                      98   BIC:                            -180.7
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   1195.1467     96.408     12.397      0.000    1003.828    1386.466
x             -0.1951      0.096     -2.024      0.046      -0.386      -0.004
==============================================================================
Omnibus:                        5.816   Durbin-Watson:                   1.889
Prob(Omnibus):                  0.055   Jarque-Bera (JB):                2.719
Skew:                           0.069   Prob(JB):                        0.257
Kurtosis:                       2.204   Cond. No.                     1.02e+07
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.02e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

regressor = smf.ols(formula = "x~y", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      x   R-squared:                       0.040
Model:                            OLS   Adj. R-squared:                  0.030
Method:                 Least Squares   F-statistic:                     4.097
Date:                Fri, 23 Feb 2018   Prob (F-statistic):             0.0457
Time:                        12:32:07   Log-Likelihood:                 92.329
No. Observations:                 100   AIC:                            -180.7
Df Residuals:                      98   BIC:                            -175.4
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   1205.6574    101.595     11.867      0.000    1004.045    1407.270
y             -0.2057      0.102     -2.024      0.046      -0.407      -0.004
==============================================================================
Omnibus:                        0.387   Durbin-Watson:                   2.083
Prob(Omnibus):                  0.824   Jarque-Bera (JB):                0.548
Skew:                           0.106   Prob(JB):                        0.760
Kurtosis:                       2.707   Cond. No.                     1.05e+07
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.05e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
"""