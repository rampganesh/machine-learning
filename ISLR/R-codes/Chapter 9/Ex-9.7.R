library(e1071)
library(ggplot2)
library(ISLR)

head(Auto)

summary(Auto$mpg)

# median : 22.75

Auto$y = ifelse(Auto$mpg > 22.75, 1, 0)

cost = c(0.1, 1, 5, 10, 25, 50, 75, 100, 500, 1000)

## Fitting linear classifier

svmfit = tune(svm, y~., data = Auto[,-c(1,9)], kernel = 'linear', ranges = list(cost = cost))

summary(svmfit)

svmfit$best.parameters

svmfit$best.performance

# Increasing the cost results in increase of error.

cost = c(0.1, 0.05, 0.025, 0.001, 0.0001)


svmfit = tune(svm, y~., data = Auto[,-c(1,9)], kernel = 'linear', ranges = list(cost = cost))

summary(svmfit)

svmfit$best.parameters

svmfit$best.performance

# Cost of 0.025 gives the best performance as the missclassifications increase for further decrease in cost

linear.svm = svm(y~., data = Auto[,-c(1,9)], kernel = 'linear', cost = 0.025)



## Radial kernel

gamma = c(0.01, 0.001, 0.1, 0.5, 1, 2, 3, 4, 5, 10)

svmfit = tune(svm, y~., data = Auto[,-c(1,9)], kernel = 'radial', ranges = list(cost = cost, gamma = gamma))

summary(svmfit)

svmfit$best.parameters

svmfit$best.performance

# Cost:0.1 and gamma : 0.5

radial.svm = svm(y~., data = Auto[,-c(1,9)], kernel = 'radial', cost = 0.1, gamma = 0.5)

## Polynomial kernel

degree = c(0.01, 0.001, 0.1, 0.5, 1, 2, 3, 4, 5, 10)

svmfit = tune(svm, y~., data = Auto[,-c(1,9)], kernel = 'polynomial', ranges = list(cost = cost, degree = degree))

summary(svmfit)

svmfit$best.parameters

svmfit$best.performance

# cost : 0.025 and degree : 1

poly.svm = svmfit$best.model



### Plotting the classifications

plot(poly.svm, Auto, weight~horsepower)
