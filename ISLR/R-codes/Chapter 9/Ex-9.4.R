library(ISLR)
library(e1071)
library(ggplot2)
library(ROCR)

set.seed(5)

x = matrix(rnorm(200*2), ncol = 2)

y = c(rep(-1, 100), rep(1, 100))

x[y == 1,] = x[y == 1,] + 3.89

dat = data.frame(x=x, y=as.factor(y))

plot(x, col = (3-y))

ggplot(data.frame(x, y), aes(X2,X1)) + geom_point(shape = 1, aes(color = factor(y)))


svmfit = svm(y~., data = dat, kernel = 'linear', cost = 10, scale = F)

plot(svmfit, dat)

# radial

svmfit = svm(y~., data = dat, kernel = 'radial', cost = 10, scale = F, gamma = 1)

plot(svmfit, dat)

# Lot of misscalssification


## Polynomial

svmfit = svm(y~., data = dat, kernel = 'polynomial', cost = 100, scale = F, gamma = 2)

plot(svmfit, dat)


# Appears to be the best

## Testing the svmfit

set.seed(5)

x = matrix(rnorm(150*2), ncol = 2)

y = c(rep(-1, 75), rep(1, 75))

x[y == 1,] = x[y == 1,] + 3.89

dat = data.frame(x=x, y=as.factor(y))

test = dat[c(51:75,126:150),]

plot(x, col = (3-y))