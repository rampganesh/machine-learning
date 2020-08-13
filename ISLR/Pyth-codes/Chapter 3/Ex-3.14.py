# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:25:14 2018

@author: julio47
"""

import pandas as pd
import numpy as np
from statsmodels.formula import api as smf
#from statsmodels.sandbox.regression.predstd import 
#from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
from statsmodels import api as sm
import random
  

random.seed(1)

x1 = np.random.uniform(size = 100)

x2 = 0.5 * x1 + (np.random.normal(100)/10)

y = 2 + 2 * x1 + 0.3 * x2 + np.random.normal(100)

pearsonr(x1,x2)

plt.scatter(x1, x2)


d = {'x1' : x1, 'x2' : x2, 'y' : y}

dataset = pd.DataFrame(d)


regressor = smf.ols(formula = "y~x1+x2", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.000
Model:                            OLS   Adj. R-squared:                 -0.010
Method:                 Least Squares   F-statistic:                   0.03508
Date:                Mon, 26 Feb 2018   Prob (F-statistic):              0.852
Time:                        19:43:47   Log-Likelihood:                -99.354
No. Observations:                 100   AIC:                             202.7
Df Residuals:                      98   BIC:                             207.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      1.2800      0.011    111.345      0.000         1.257     1.303
x1            -5.1246      0.214    -23.964      0.000        -5.549    -4.700
x2            10.3278      0.011    961.286      0.000        10.306    10.349
==============================================================================
Omnibus:                       79.037   Durbin-Watson:                   2.075
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                7.782
Skew:                          -0.027   Prob(JB):                       0.0204
Kurtosis:                       1.634   Cond. No.                     6.15e+16
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 2.84e-30. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
"""

regressor = smf.ols(formula = "y~x1", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.000
Model:                            OLS   Adj. R-squared:                 -0.010
Method:                 Least Squares   F-statistic:                   0.03508
Date:                Mon, 26 Feb 2018   Prob (F-statistic):              0.852
Time:                        19:44:43   Log-Likelihood:                -99.354
No. Observations:                 100   AIC:                             202.7
Df Residuals:                      98   BIC:                             207.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept    105.2836      0.118    892.279      0.000       105.049   105.518
x1             0.0392      0.210      0.187      0.852        -0.377     0.455
==============================================================================
Omnibus:                       79.037   Durbin-Watson:                   2.075
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                7.782
Skew:                          -0.027   Prob(JB):                       0.0204
Kurtosis:                       1.634   Cond. No.                         3.93
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

regressor = smf.ols(formula = "y~x2", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.000
Model:                            OLS   Adj. R-squared:                 -0.010
Method:                 Least Squares   F-statistic:                   0.03508
Date:                Mon, 26 Feb 2018   Prob (F-statistic):              0.852
Time:                        19:45:19   Log-Likelihood:                -99.354
No. Observations:                 100   AIC:                             202.7
Df Residuals:                      98   BIC:                             207.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept    104.4932      4.318     24.197      0.000        95.923   113.063
x2             0.0785      0.419      0.187      0.852        -0.753     0.910
==============================================================================
Omnibus:                       79.037   Durbin-Watson:                   2.075
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                7.782
Skew:                          -0.027   Prob(JB):                       0.0204
Kurtosis:                       1.634   Cond. No.                         680.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

d = {'x1' : 0.1, 'x2' : 0.8, 'y' : 6}

dataset = dataset.append(d, ignore_index = True)


regressor = smf.ols(formula = "y~x1+x2", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.996
Model:                            OLS   Adj. R-squared:                  0.996
Method:                 Least Squares   F-statistic:                 1.120e+04
Date:                Mon, 26 Feb 2018   Prob (F-statistic):          2.05e-116
Time:                        19:50:04   Log-Likelihood:                -99.845
No. Observations:                 101   AIC:                             205.7
Df Residuals:                      98   BIC:                             213.5
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -1.9936      0.713     -2.794      0.006        -3.409    -0.578
x1            -5.2872      0.217    -24.410      0.000        -5.717    -4.857
x2            10.6528      0.072    148.663      0.000        10.511    10.795
==============================================================================
Omnibus:                       73.137   Durbin-Watson:                   2.076
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                7.673
Skew:                          -0.027   Prob(JB):                       0.0216
Kurtosis:                       1.651   Cond. No.                         113.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


regressor = smf.ols(formula = "y~x1", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.013
Model:                            OLS   Adj. R-squared:                  0.004
Method:                 Least Squares   F-statistic:                     1.352
Date:                Mon, 26 Feb 2018   Prob (F-statistic):              0.248
Time:                        19:50:35   Log-Likelihood:                -373.70
No. Observations:                 101   AIC:                             751.4
Df Residuals:                      99   BIC:                             756.6
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept    102.6407      1.747     58.762      0.000        99.175   106.107
x1             3.6234      3.117      1.163      0.248        -2.561     9.808
==============================================================================
Omnibus:                      215.003   Durbin-Watson:                   1.050
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            37615.437
Skew:                          -9.635   Prob(JB):                         0.00
Kurtosis:                      95.558   Cond. No.                         3.91
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


regressor = smf.ols(formula = "y~x2", data = dataset).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.969
Model:                            OLS   Adj. R-squared:                  0.969
Method:                 Least Squares   F-statistic:                     3112.
Date:                Mon, 26 Feb 2018   Prob (F-statistic):           1.31e-76
Time:                        19:50:57   Log-Likelihood:                -198.69
No. Observations:                 101   AIC:                             401.4
Df Residuals:                      99   BIC:                             406.6
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      0.4995      1.869      0.267      0.790        -3.209     4.209
x2            10.1688      0.182     55.783      0.000         9.807    10.531
==============================================================================
Omnibus:                       10.646   Durbin-Watson:                   2.055
Prob(Omnibus):                  0.005   Jarque-Bera (JB):                3.670
Skew:                          -0.002   Prob(JB):                        0.160
Kurtosis:                       2.066   Cond. No.                         111.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

sm.graphics.influence_plot(regressor, criterion = 'Cooks', size = 2, alpha = 0.05)
