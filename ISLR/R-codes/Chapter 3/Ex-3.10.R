library(ISLR)

dataset = Carseats[,c(1,6,10:11)]

# dataset$Urban = factor(dataset$Urban, levels = c("Yes","No"), labels = c(1,0))
# 
# dataset$US = factor(dataset$US, levels = c("Yes","No"), labels = c(1,0))

lm.fit = lm(formula = "Sales ~ .", data = dataset)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -6.9206 -1.6220 -0.0564  1.5786  7.0581 
# 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13.043469   0.651012  20.036  < 2e-16 ***
# Price       -0.054459   0.005242 -10.389  < 2e-16 ***
# UrbanYes    -0.021916   0.271650  -0.081    0.936    
# USYes        1.200573   0.259042   4.635 4.86e-06 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.472 on 396 degrees of freedom
# Multiple R-squared:  0.2393,	Adjusted R-squared:  0.2335 
# F-statistic: 41.52 on 3 and 396 DF,  p-value: < 2.2e-16

# Removing the statistically insignificant vars

lm.fit = lm(formula = "Sales ~ Price+US", data = dataset)
summary(lm.fit)


# Residuals:
#   Min      1Q  Median      3Q     Max 
# -6.9269 -1.6286 -0.0574  1.5766  7.0515 
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13.03079    0.63098  20.652  < 2e-16 ***
# Price       -0.05448    0.00523 -10.416  < 2e-16 ***
# USYes        1.19964    0.25846   4.641 4.71e-06 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.469 on 397 degrees of freedom
# Multiple R-squared:  0.2393,	Adjusted R-squared:  0.2354 
# F-statistic: 62.43 on 2 and 397 DF,  p-value: < 2.2e-16

# Improved R-squared value in the second model meaning it is capturing 
# more of the variation between the variables


# Confidence intervals

confint(lm.fit)

#                     2.5 %      97.5 %
# (Intercept) 11.79032020 14.27126531
# Price       -0.06475984 -0.04419543
# USYes        0.69151957  1.70776632


# Outliers and high leverage points

plot(lm.fit)

plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

# No evidence of outliers as all values between -3 and 3
# Obs 43 appears to be a high leverage point