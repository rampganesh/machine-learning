library(ISLR)

pairs(mpg ~ .-name, data = Auto)

# Correlation matrix

cor(Auto[,-9],)

#                     mpg  cylinders displacement horsepower     weight acceleration       year     origin
# mpg           1.0000000 -0.7776175   -0.8051269 -0.7784268 -0.8322442    0.4233285  0.5805410  0.5652088
# cylinders    -0.7776175  1.0000000    0.9508233  0.8429834  0.8975273   -0.5046834 -0.3456474 -0.5689316
# displacement -0.8051269  0.9508233    1.0000000  0.8972570  0.9329944   -0.5438005 -0.3698552 -0.6145351
# horsepower   -0.7784268  0.8429834    0.8972570  1.0000000  0.8645377   -0.6891955 -0.4163615 -0.4551715
# weight       -0.8322442  0.8975273    0.9329944  0.8645377  1.0000000   -0.4168392 -0.3091199 -0.5850054
# acceleration  0.4233285 -0.5046834   -0.5438005 -0.6891955 -0.4168392    1.0000000  0.2903161  0.2127458
# year          0.5805410 -0.3456474   -0.3698552 -0.4163615 -0.3091199    0.2903161  1.0000000  0.1815277
# origin        0.5652088 -0.5689316   -0.6145351 -0.4551715 -0.5850054    0.2127458  0.1815277  1.0000000

# regressor fit

lm.fit = lm(formula = "mpg~.-name", data = Auto)
summary(lm.fit)


# Residuals:
#   Min      1Q  Median      3Q     Max 
# -9.5903 -2.1565 -0.1169  1.8690 13.0604 
# 
# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -17.218435   4.644294  -3.707  0.00024 ***
# cylinders     -0.493376   0.323282  -1.526  0.12780    
# displacement   0.019896   0.007515   2.647  0.00844 ** 
# horsepower    -0.016951   0.013787  -1.230  0.21963    
# weight        -0.006474   0.000652  -9.929  < 2e-16 ***
# acceleration   0.080576   0.098845   0.815  0.41548    
# year           0.750773   0.050973  14.729  < 2e-16 ***
# origin         1.426141   0.278136   5.127 4.67e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 3.328 on 384 degrees of freedom
# Multiple R-squared:  0.8215,	Adjusted R-squared:  0.8182 
# F-statistic: 252.4 on 7 and 384 DF,  p-value: < 2.2e-16


# Diagnostic plots

plot(lm.fit)

# interaction between vars

lm.fit = lm("mpg ~ displacement+weight+year*origin", data = Auto[,-9])
summary(lm.fit)

lm.fit = lm("mpg ~ displacement+year+weight*origin", data = Auto[,-9])
summary(lm.fit)

lm.fit = lm("mpg ~ displacement*weight+year+origin", data = Auto[,-9])
summary(lm.fit)


# Polynomial and log effects


lm.fit = lm("mpg ~ displacement+I(log(weight))+year+origin", data = Auto[,-9])
summary(lm.fit)

lm.fit = lm("mpg ~ I(log(displacement))+year+weight+origin", data = Auto[,-9])
summary(lm.fit)

lm.fit = lm("mpg ~ displacement+weight+year+I(origin^2)", data = Auto[,-9])
summary(lm.fit)

lm.fit = lm("mpg ~ displacement+I(sqrt(weight))+year+origin", data = Auto[,-9])
summary(lm.fit)

lm.fit = lm("mpg ~ I(displacement^2)+weight+year+origin", data = Auto[,-9])
summary(lm.fit)

# seems to have the most effect
