library(ISLR)

Wage

attach(Wage)


agelims = range(age)

grid = seq(agelims[1], agelims[2])

fit.1= lm(wage~age, data = Wage)
fit.2= lm(wage~poly(age,2), data = Wage)
fit.3= lm(wage~poly(age,3), data = Wage)
fit.4= lm(wage~poly(age,4), data = Wage)
fit.5= lm(wage~poly(age,5), data = Wage)
fit.6= lm(wage~poly(age,6), data = Wage)
fit.7= lm(wage~poly(age,7), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5, fit.6, fit.7)

set.seed(1)

library(boot)
cv.error.10 = rep(0, 10)
for ( i in 1:10 ) {
  fit = glm(wage ~ poly(age, i), data = Wage)
  cv.error.10[i] = cv.glm(Wage, fit, K = 10)$delta[1]
}
plot(cv.error.10, type = "l", lwd = 2, col = "blue", xlab = "Degree of Polynomial", ylab = "Test Error")
title("Cross-validation with K = 10")
points(which.min(cv.error.10), min(cv.error.10), col = "red", cex = 2, pch = 20)


# The above can also be written as
set.seed(1)

k = 10; p = 10

folds = sample(1:k, length(age), replace = T)

errors = matrix(NA, k, p)

for ( i in 1:k ) {
  for ( j in 1:p ) {
    fit = lm(wage ~ poly(age, j), subset = (folds != i))
    preds = predict(fit, newdata = Wage[folds == i, ])
    errors[i, j] = mean((preds - wage[folds == i])^2)
  }
}

errors = apply(errors, 2, mean)

which.min(errors)


######################################################################

set.seed(1)

# cross-validation
cv.error <- rep(0,9)
for (i in 2:10) {
  Wage$age.cut <- cut(Wage$age,i)
  glm.fit <- glm(wage~age.cut, data=Wage)
  cv.error[i-1] <- cv.glm(Wage, glm.fit, K=10)$delta[1]  # [1]:std, [2]:bias-corrected
}
cv.error

plot(2:10, cv.error, type = "l")


# Fitting the best option for cut

fit = glm(wage~cut(age,8), data = Wage)
preds = predict(fit, newdata = list(age = grid), se = T)

se.bands = cbind(preds$fit+ 2*preds$se.fit, preds$fit - 2*preds$se.fit)

plot(age, wage, col = "lightgrey")

lines(grid, preds$fit, col = "blue")

matlines(grid, se.bands, col = "red")