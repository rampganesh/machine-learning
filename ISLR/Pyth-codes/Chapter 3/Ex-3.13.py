# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:11:25 2018

@author: mzayauser
"""


import pandas as pd
import numpy as np
from statsmodels.formula import api as smf
#from statsmodels.sandbox.regression.predstd import 
#from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import random

random.seed(1)

x = np.random.normal(0, 1, 100)

eps =  np.random.normal(0, 0.25, 100)

y = -1 + 0.5 * x + eps

plt.scatter(x, y)

# Linear relationship is highly probable



d = {'x' : x, 'y' : y}

dataset = pd.DataFrame(d)


regressor =  smf.ols(formula = "y~x", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.776
Model:                            OLS   Adj. R-squared:                  0.774
Method:                 Least Squares   F-statistic:                     339.7
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           1.29e-33
Time:                        18:04:18   Log-Likelihood:                -4.3966
No. Observations:                 100   AIC:                             12.79
Df Residuals:                      98   BIC:                             18.00
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -1.0195      0.026    -39.874      0.000      -1.070      -0.969
x              0.5366      0.029     18.432      0.000       0.479       0.594
==============================================================================
Omnibus:                        5.112   Durbin-Watson:                   1.669
Prob(Omnibus):                  0.078   Jarque-Bera (JB):                4.577
Skew:                          -0.510   Prob(JB):                        0.101
Kurtosis:                       3.237   Cond. No.                         1.15
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

#x_reg = np.concatenate((np.array([0]), x))
#regression_line = np.concatenate((np.array([regressor.params[0]]), regressor.predict()))

plt.scatter(x, y)
plt.plot(x_reg, regressor.predict(), '-', color = 'red')


regressorp = smf.ols(formula = "y~x+np.power(x,2)", data = dataset).fit()
regressorp.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.777
Model:                            OLS   Adj. R-squared:                  0.772
Method:                 Least Squares   F-statistic:                     168.9
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           2.55e-32
Time:                        18:31:06   Log-Likelihood:                -4.2312
No. Observations:                 100   AIC:                             14.46
Df Residuals:                      97   BIC:                             22.28
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -1.0084      0.032    -31.293      0.000      -1.072      -0.944
x                  0.5310      0.031     17.195      0.000       0.470       0.592
np.power(x, 2)    -0.0146      0.026     -0.567      0.572      -0.066       0.037
==============================================================================
Omnibus:                        5.494   Durbin-Watson:                   1.686
Prob(Omnibus):                  0.064   Jarque-Bera (JB):                4.979
Skew:                          -0.533   Prob(JB):                       0.0830
Kurtosis:                       3.244   Cond. No.                         2.29
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


#************************************ Reducing variance on error ****************************

eps =  np.random.normal(0, 0.1, 100)

y = -1 + 0.5 * x + eps

plt.scatter(x, y)

# Points are closer together


d = {'x' : x, 'y' : y}

dataset = pd.DataFrame(d)


regressor1 =  smf.ols(formula = "y~x", data = dataset).fit()

regressor1.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.960
Model:                            OLS   Adj. R-squared:                  0.959
Method:                 Least Squares   F-statistic:                     2333.
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           3.81e-70
Time:                        18:33:50   Log-Likelihood:                 98.976
No. Observations:                 100   AIC:                            -194.0
Df Residuals:                      98   BIC:                            -188.7
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.9989      0.009   -109.839      0.000      -1.017      -0.981
x              0.5002      0.010     48.300      0.000       0.480       0.521
==============================================================================
Omnibus:                        4.753   Durbin-Watson:                   2.119
Prob(Omnibus):                  0.093   Jarque-Bera (JB):                4.807
Skew:                          -0.513   Prob(JB):                       0.0904
Kurtosis:                       2.684   Cond. No.                         1.15
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Reducing the error has drastically improved the fit

plt.scatter(x, y)
plt.plot(x, regressor1.predict(), '-', color = 'red')


regressorp1 = smf.ols(formula = "y~x+np.power(x,2)", data = dataset).fit()
regressorp1.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.960
Model:                            OLS   Adj. R-squared:                  0.959
Method:                 Least Squares   F-statistic:                     1155.
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           2.31e-68
Time:                        18:35:40   Log-Likelihood:                 98.978
No. Observations:                 100   AIC:                            -192.0
Df Residuals:                      97   BIC:                            -184.1
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -0.9984      0.011    -86.968      0.000      -1.021      -0.976
x                  0.4999      0.011     45.446      0.000       0.478       0.522
np.power(x, 2)    -0.0006      0.009     -0.062      0.951      -0.019       0.018
==============================================================================
Omnibus:                        4.751   Durbin-Watson:                   2.120
Prob(Omnibus):                  0.093   Jarque-Bera (JB):                4.807
Skew:                          -0.514   Prob(JB):                       0.0904
Kurtosis:                       2.690   Cond. No.                         2.29
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""




# ********************************** Increasing the variance of error*******************************

eps =  np.random.normal(0, 0.75, 100)

y = -1 + 0.5 * x + eps

plt.scatter(x, y)

# Less likelihood of linear relationship


d = {'x' : x, 'y' : y}

dataset = pd.DataFrame(d)


regressor2 =  smf.ols(formula = "y~x", data = dataset).fit()

regressor2.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.297
Model:                            OLS   Adj. R-squared:                  0.290
Method:                 Least Squares   F-statistic:                     41.47
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           4.46e-09
Time:                        18:39:56   Log-Likelihood:                -100.35
No. Observations:                 100   AIC:                             204.7
Df Residuals:                      98   BIC:                             209.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.9515      0.067    -14.255      0.000      -1.084      -0.819
x              0.4894      0.076      6.440      0.000       0.339       0.640
==============================================================================
Omnibus:                        0.746   Durbin-Watson:                   2.156
Prob(Omnibus):                  0.689   Jarque-Bera (JB):                0.688
Skew:                          -0.198   Prob(JB):                        0.709
Kurtosis:                       2.912   Cond. No.                         1.15
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Increasing the error has resulted in a model that only exhibits 30% relationship between x and y

plt.scatter(x, y)
plt.plot(x, regressor2.predict(), '-', color = 'red')


regressorp2 = smf.ols(formula = "y~x+np.power(x,2)", data = dataset).fit()
regressorp2.summary()


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.298
Model:                            OLS   Adj. R-squared:                  0.283
Method:                 Least Squares   F-statistic:                     20.57
Date:                Fri, 23 Feb 2018   Prob (F-statistic):           3.56e-08
Time:                        18:41:51   Log-Likelihood:                -100.32
No. Observations:                 100   AIC:                             206.6
Df Residuals:                      97   BIC:                             214.4
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -0.9379      0.084    -11.135      0.000      -1.105      -0.771
x                  0.4825      0.081      5.978      0.000       0.322       0.643
np.power(x, 2)    -0.0179      0.067     -0.266      0.791      -0.152       0.116
==============================================================================
Omnibus:                        0.772   Durbin-Watson:                   2.158
Prob(Omnibus):                  0.680   Jarque-Bera (JB):                0.623
Skew:                          -0.193   Prob(JB):                        0.732
Kurtosis:                       2.981   Cond. No.                         2.29
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""