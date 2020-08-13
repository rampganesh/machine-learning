set.seed(1)

x = rnorm(100)

y = 2 * x + rnorm(100)


dataset = data.frame(x = x, y = y)

lm.fit = lm(formula = "y~x+0", data = dataset)
summary(lm.fit)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -1.9154 -0.6472 -0.1771  0.5056  2.3109 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# x   1.9939     0.1065   18.73   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.9586 on 99 degrees of freedom
# Multiple R-squared:  0.7798,	Adjusted R-squared:  0.7776 
# F-statistic: 350.7 on 1 and 99 DF,  p-value: < 2.2e-16

lm.fit1 = lm(formula = "x~y+0", data = dataset)
summary(lm.fit1)


# Residuals:
#     Min      1Q  Median      3Q     Max 
# -0.8699 -0.2368  0.1030  0.2858  0.8938 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# y  0.39111    0.02089   18.73   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.4246 on 99 degrees of freedom
# Multiple R-squared:  0.7798,	Adjusted R-squared:  0.7776 
# F-statistic: 350.7 on 1 and 99 DF,  p-value: < 2.2e-16


y_d = predict(lm.fit)
x_d = predict(lm.fit1)

num = (sqrt(99)* sum(x+y))

den = sqrt((sum((x)^2) * sum(y^2) -  sum(x*y)^2))
