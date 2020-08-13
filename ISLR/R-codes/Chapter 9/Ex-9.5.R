library(ISLR)
library(e1071)
library(ggplot2)

set.seed(1)

x1 = runif(500) - 0.5

x2 = runif(500) - 0.5

y = 1*(x1^2 - x2^2 > 0)

ggplot(data.frame(x1,x2,y), aes(x1,x2)) + geom_point(shape=1, aes(color=factor(y)))

glmfit = glm(y ~ x1+x2, family = binomial)

summary(glmfit)


## Predict the training data

preds = predict(glmfit, type = 'response')

preds = ifelse(preds > 0.5, 1, 0)

ggplot(data.frame(x1,x2,preds), aes(x1,x2)) + geom_point(shape=1, aes(color=factor(preds)))


# It is linear alright

## Mapping interactions between predictors

## x1^2

glmfit = glm(y ~ poly(x1,2)+x2, family = binomial)

summary(glmfit)

## Predict and plot

preds = predict(glmfit, type = 'response')

preds = ifelse(preds > 0.5, 1, 0)

ggplot(data.frame(x1,x2,preds), aes(x1,x2)) + geom_point(shape=1, aes(color=factor(preds)))

# Still linear


### X1*x2


glmfit = glm(y ~ x1+x2+x1*x2, family = binomial)

summary(glmfit)

## Predict and plot

preds = predict(glmfit, type = 'response')

preds = ifelse(preds > 0.5, 1, 0)

ggplot(data.frame(x1,x2,preds), aes(x1,x2)) + geom_point(shape=1, aes(color=factor(preds)))


### log


glmfit = glm(y ~ x1+x2+poly(x1,3), family = binomial)

summary(glmfit)

## Predict and plot

preds = predict(glmfit, type = 'response')

preds = ifelse(preds > 0.5, 1, 0)

ggplot(data.frame(x1,x2,preds), aes(x1,x2)) + geom_point(shape=1, aes(color=factor(preds)))


### poly again


glmfit = glm(y ~ poly(x1,2)+poly(x2,2), family = binomial)

summary(glmfit)

## Predict and plot

preds = predict(glmfit, type = 'response')

preds = ifelse(preds > 0.5, 1, 0)

ggplot(data.frame(x1,x2,preds), aes(x1,x2)) + geom_point(shape=1, aes(color=factor(preds)))


## Seems to fit it perfectly


#### SVC to the data

dat = data.frame(x1, x2, y = factor(y))

svmfit  = svm(y~., kernel = 'linear', data = dat)

plot(svmfit, data = dat)

# Linear classification boundary


## Radial kernel


dat = data.frame(x1, x2, y = factor(y))

svmfit  = svm(y~., kernel = 'radial', data = dat, gamma = 1, cost = 10)

plot(svmfit, data = dat)

# this performs better

svmfit  = svm(y~., kernel = 'radial', data = dat, gamma = 2, cost = 100)

plot(svmfit, data = dat)


# Logistic regression after using polynomial transformation gives the same results as using a svc with radial 
# kernel
