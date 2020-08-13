# Auto Dataset

library(ISLR)

lm.fit = lm(mpg~horsepower, data = Auto)

summary(lm.fit)

# Call:
#   lm(formula = mpg ~ horsepower, data = Auto)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -13.5710  -3.2592  -0.3435   2.7630  16.9240 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 39.935861   0.717499   55.66   <2e-16 ***
#   horsepower  -0.157845   0.006446  -24.49   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.906 on 390 degrees of freedom
# Multiple R-squared:  0.6059,	Adjusted R-squared:  0.6049 
# F-statistic: 599.7 on 1 and 390 DF,  p-value: < 2.2e-16

predict(lm.fit, newdata = data.frame(horsepower = 98))

# 1 
# 24.46708 

predict(lm.fit, newdata = data.frame(horsepower = 98), interval = "confidence")
 
# fit      lwr      upr
# 1 24.46708 23.97308 24.96108

predict(lm.fit, newdata = data.frame(horsepower = 98), interval = "prediction")

# fit     lwr      upr
# 1 24.46708 14.8094 34.12476

plot(Auto$horsepower,Auto$mpg, xlab = "Horsepower", ylab = "MPG")
abline(lm.fit, col = "red")

plot(lm.fit)