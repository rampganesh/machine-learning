# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:12:38 2018

@author: julio47
"""

from sklearn.linear_model import LinearRegression
from statsmodels.formula import api as smf
from statsmodels import api as sm
#from statsmodels.sandbox.regression.predstd import 
from statsmodels.regression import linear_model
import pandas as pd
#from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import os
import math
import matplotlib


filename = 'Auto.csv'

Auto = pd.read_csv(filename)

#regressor = LinearRegression()

#X = Auto["horsepower"]
#
#y = Auto["mpg"]
#
##regressor.fit(X.reshape(-1, 1), y.reshape(-1, 1))

regressor = smf.ols(formula='mpg ~ horsepower', data=Auto).fit()

regressor.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.606
Model:                            OLS   Adj. R-squared:                  0.605
Method:                 Least Squares   F-statistic:                     599.7
Date:                Fri, 09 Feb 2018   Prob (F-statistic):           7.03e-81
Time:                        23:08:07   Log-Likelihood:                -1178.7
No. Observations:                 392   AIC:                             2361.
Df Residuals:                     390   BIC:                             2369.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     39.9359      0.717     55.660      0.000        38.525    41.347
horsepower    -0.1578      0.006    -24.489      0.000        -0.171    -0.145
==============================================================================
Omnibus:                       16.432   Durbin-Watson:                   0.920
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               17.305
Skew:                           0.492   Prob(JB):                     0.000175
Kurtosis:                       3.299   Cond. No.                         322.
==============================================================================
"""

d = {'horsepower' : pd.Series([98])}

y_pred= pd.DataFrame(d)

regressor.predict(y_pred)

#array([ 24.46707715])


regressor = linear_model.OLS(y,X)


regressor.model.predict(regressor.params, np.array([[98],[98]]))

#array([ 24.46707715])

#prstd, iv_l, iv_u = wls_prediction_std(regressor, exog = np.array([[98],[98]]),  alpha = 0.05)

#predvar = res.mse_resid/weights + (exog * np.dot(covb, exog.T).T).sum(1)
#    predstd = np.sqrt(predvar)
#    tppf = stats.t.isf(alpha/2., res.df_resid)
#    interval_u = predicted + tppf * predstd
#interval_l = predicted - tppf * predstd
#
#
#regressor.get_prediction(y_pred)

plt.scatter(Auto["horsepower"], Auto["mpg"], color = 'red')
plt.plot(Auto["horsepower"], regressor.predict(Auto["horsepower"]), color = 'blue')
plt.title('MPG vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()



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
sm.graphics.influence_plot(lm, criterion = 'Cooks', size = 2, ax = ax4)

plt.tight_layout()
fig.savefig('regplots.png')
