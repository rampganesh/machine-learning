set.seed(1)

x = rnorm(mean = 0, sd = 1, n = 100)

eps = rnorm(mean = 0, sd = 0.25, n = 100)

y = -1 + 0.5 * x + eps



plot(x, y)

# Strong evidence of linear relationship


lm.fit = lm(y~x)

lm.fit$coefficients

# (Intercept)          x 
# -1.0094232   0.4997349

abline(lm.fit, col = "blue")

summary(lm.fit)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -0.46921 -0.15344 -0.03487  0.13485  0.58654 
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -1.00942    0.02425  -41.63   <2e-16 ***
# x            0.49973    0.02693   18.56   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.2407 on 98 degrees of freedom
# Multiple R-squared:  0.7784,	Adjusted R-squared:  0.7762 
# F-statistic: 344.3 on 1 and 98 DF,  p-value: < 2.2e-16

dataset = data.frame(x = x, y = y)

lm.fitp = lm("y~x+I(x^2)", data = dataset)

summary(lm.fitp)

# Residuals:
#   Min      1Q  Median      3Q     Max 
# -0.4913 -0.1563 -0.0322  0.1451  0.5675 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.98582    0.02941 -33.516   <2e-16 ***
#   x            0.50429    0.02700  18.680   <2e-16 ***
#   I(x^2)      -0.02973    0.02119  -1.403    0.164    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.2395 on 97 degrees of freedom
# Multiple R-squared:  0.7828,	Adjusted R-squared:  0.7784 
# F-statistic: 174.8 on 2 and 97 DF,  p-value: < 2.2e-16


# ************************** Decreasing variance of error term ****************


eps = rnorm(mean = 0, sd = 0.1, n = 100)

y = -1 + 0.5 * x + eps

dataset = data.frame(x = x, y = y)

lm.fit1 = lm("y~x", data = dataset)


lm.fit1$coefficients

# (Intercept)           x 
#  -0.9972631   0.5021167 


plot(x, y)

# Strong evidence of linear relationship

abline(lm.fit1, col = "blue")

summary(lm.fit1)

# Residuals:
#       Min        1Q    Median        3Q       Max 
# -0.291411 -0.048230 -0.004533  0.064924  0.264157 
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.99726    0.01047  -95.25   <2e-16 ***
# x            0.50212    0.01163   43.17   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.1039 on 98 degrees of freedom
# Multiple R-squared:  0.9501,	Adjusted R-squared:  0.9495 
# F-statistic:  1864 on 1 and 98 DF,  p-value: < 2.2e-16

# Reducing the error drastically improves the model and ensures a linear model.

lm.fitp1 = lm("y~x+I(x^2)", data = dataset)

summary(lm.fitp1)

# Residuals:
#   Min        1Q    Median        3Q       Max 
# -0.292742 -0.049523 -0.003585  0.065955  0.264542 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.995887   0.012828 -77.637   <2e-16 ***
#   x            0.502382   0.011773  42.673   <2e-16 ***
#   I(x^2)      -0.001734   0.009242  -0.188    0.852    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.1045 on 97 degrees of freedom
# Multiple R-squared:  0.9501,	Adjusted R-squared:  0.949 
# F-statistic: 922.9 on 2 and 97 DF,  p-value: < 2.2e-16

# Not much difference


#************************** Increasing the variance of the error term ***************


eps = rnorm(mean = 0, sd = 0.75, n = 100)

y = -1 + 0.5 * x + eps

dataset = data.frame(x = x, y = y)

lm.fit2 = lm("y~x", data = dataset)


lm.fit2$coefficients

# (Intercept)           x 
# -0.9567510   0.4582355 


plot(x, y)

# Does not look that linear


abline(lm.fit2, col = "blue")

summary(lm.fit2)

# Residuals:
#   Min       1Q   Median       3Q      Max 
# -1.88719 -0.40893 -0.02832  0.50466  1.40916 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.95675    0.07521 -12.721  < 2e-16 ***
#   x            0.45824    0.08354   5.485 3.23e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.7466 on 98 degrees of freedom
# Multiple R-squared:  0.2349,	Adjusted R-squared:  0.2271 
# F-statistic: 30.09 on 1 and 98 DF,  p-value: 3.227e-07

# Barely a linear model due to increase in variance.

lm.fitp2 = lm("y~x+I(x^2)", data = dataset)

summary(lm.fitp2)


# Residuals:
#   Min       1Q   Median       3Q      Max 
# -1.80107 -0.45760 -0.04531  0.52073  1.40496 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.84398    0.08996  -9.382 2.90e-15 ***
#   x            0.48000    0.08256   5.814 7.81e-08 ***
#   I(x^2)      -0.14205    0.06481  -2.192   0.0308 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.7325 on 97 degrees of freedom
# Multiple R-squared:  0.271,	Adjusted R-squared:  0.256 
# F-statistic: 18.03 on 2 and 97 DF,  p-value: 2.199e-07


# Adding the polynomial term improves the fit a little


confint(lm.fit)

#                  2.5 %     97.5 %
# (Intercept) -1.0575402 -0.9613061
# x            0.4462897  0.5531801

confint(lm.fit1)

#                  2.5 %     97.5 %
# (Intercept) -1.0180413 -0.9764850
# x            0.4790377  0.5251957

confint(lm.fit2)

#                  2.5 %     97.5 %
# (Intercept) -1.1060050 -0.8074970
# x            0.2924541  0.6240169


# Where variance is low, the interval is narrow in comparison