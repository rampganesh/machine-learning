# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:09:34 2018

@author: julio47
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

regressor = smf.ols(formula = "y~x+0", data = dataset).fit()

regressor.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.850
Model:                            OLS   Adj. R-squared:                  0.848
Method:                 Least Squares   F-statistic:                     559.8
Date:                Sun, 18 Feb 2018   Prob (F-statistic):           1.56e-42
Time:                        17:53:47   Log-Likelihood:                -127.90
No. Observations:                 100   AIC:                             257.8
Df Residuals:                      99   BIC:                             260.4
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
x              1.9820      0.084     23.661      0.000         1.816     2.148
==============================================================================
Omnibus:                        4.949   Durbin-Watson:                   1.790
Prob(Omnibus):                  0.084   Jarque-Bera (JB):                4.528
Skew:                          -0.516   Prob(JB):                        0.104
Kurtosis:                       3.141   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


regressor = smf.ols(formula = "x~y+0", data = dataset).fit()

regressor.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      x   R-squared:                       0.850
Model:                            OLS   Adj. R-squared:                  0.848
Method:                 Least Squares   F-statistic:                     559.8
Date:                Sun, 18 Feb 2018   Prob (F-statistic):           1.56e-42
Time:                        17:59:35   Log-Likelihood:                -51.351
No. Observations:                 100   AIC:                             104.7
Df Residuals:                      99   BIC:                             107.3
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
y              0.4287      0.018     23.661      0.000         0.393     0.465
==============================================================================
Omnibus:                        1.491   Durbin-Watson:                   1.862
Prob(Omnibus):                  0.474   Jarque-Bera (JB):                1.482
Skew:                           0.284   Prob(JB):                        0.477
Kurtosis:                       2.821   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""