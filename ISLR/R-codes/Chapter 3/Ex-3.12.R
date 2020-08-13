set.seed(1)

x = rnorm(100)

y = 2*x + rnorm(100)

lm.fit = lm(y~x)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -1.8768 -0.6138 -0.1395  0.5394  2.3462 
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.03769    0.09699  -0.389    0.698    
# x            1.99894    0.10773  18.556   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.9628 on 98 degrees of freedom
# Multiple R-squared:  0.7784,	Adjusted R-squared:  0.7762 
# F-statistic: 344.3 on 1 and 98 DF,  p-value: < 2.2e-16

lm.fit1 = lm(x~y)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -0.90848 -0.28101  0.06274  0.24570  0.85736 
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  0.03880    0.04266    0.91    0.365    
# y            0.38942    0.02099   18.56   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.4249 on 98 degrees of freedom
# Multiple R-squared:  0.7784,	Adjusted R-squared:  0.7762 
# F-statistic: 344.3 on 1 and 98 DF,  p-value: < 2.2e-16



set.seed(1)
x <- rnorm(100, mean=1000, sd=0.1)
y <- rnorm(100, mean=1000, sd=0.1)
lm.fit <- lm(y ~ x)
lm.fit1 <- lm(x ~ y)
summary(lm.fit)

# Residuals:
#     Min       1Q   Median       3Q      Max 
# -0.18768 -0.06138 -0.01395  0.05394  0.23462 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 1001.05662  107.72820   9.292 4.16e-15 ***
# x             -0.00106    0.10773  -0.010    0.992    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.09628 on 98 degrees of freedom
# Multiple R-squared:  9.887e-07,	Adjusted R-squared:  -0.0102 
# F-statistic: 9.689e-05 on 1 and 98 DF,  p-value: 0.9922
# 

summary(lm.fit1)

# Residuals:
#       Min        1Q    Median        3Q       Max 
# -0.232416 -0.060361  0.000536  0.058305  0.229316 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  1.001e+03  9.472e+01   10.57   <2e-16 ***
# y           -9.324e-04  9.472e-02   -0.01    0.992    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.09028 on 98 degrees of freedom
# Multiple R-squared:  9.887e-07,	Adjusted R-squared:  -0.0102 
# F-statistic: 9.689e-05 on 1 and 98 DF,  p-value: 0.9922

