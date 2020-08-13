library(e1071)
library(ggplot2)
library(ROCR)
library(ISLR)

set.seed(1)

x = matrix(rnorm(20*2), ncol = 2)

y = c(rep(-1, 10), rep(1, 10))

x[y == 1,] = x[y == 1,] + 1

dat = data.frame(x=x, y=as.factor(y))

plot(x, col = (3 - y))

# GGplot

ggplot(dat, aes(x = x.1, y = x.2)) + geom_point(aes(colour = y))

# Fitting SVC

svmfit = svm(y~., data = dat, kernel = 'linear', cost = 10, scale = F)

plot(svmfit, dat)

dat1 = dat

dat1$fitted = svmfit$fitted

dat1$shape1 = 1

dat1$shape1[svmfit$index] = 2

dat1$shape1 = as.factor(dat1$shape)
  
ggplot(dat1, aes(x = x.2, y = x.1))+geom_point(aes(colour = fitted, shape = shape1), size = 3)+geom_abline(intercept = -1.5)


# Smaller value of cost

svmfit = svm(y~., data = dat, kernel = 'linear', cost = 0.1, scale = F)

plot(svmfit, dat)


# Tuning

set.seed(1)

tune.out = tune(svm, y~., data = dat, kernel = 'linear', ranges = list(cost = c(10^seq(-3,2),5)))

summary(tune.out)

bestmod = tune.out$best.model

summary(bestmod)

# Test data set


xtest = matrix(rnorm(20*2), ncol = 2)

ytest = sample(c(-1, 1), 20, rep = T)

xtest[ytest == 1,] = xtest[ytest == 1,] + 1

testdat = data.frame(x=xtest, y=as.factor(ytest))

ypred = predict(bestmod, testdat)

table(predict = ypred, truth = testdat$y)

# 19 predicted correctly

# using 0.01 as cost

svmfit = svm(y~., data = dat, kernel = 'linear', cost = 0.01, scale = F)

ypred = predict(svmfit, testdat)

table(predict = ypred, truth = testdat$y)

# Making data linearly inseparable

x[y == 1,] = x[y == 1,] + 0.5

plot(x, col =(y + 5)/2, pch =  19)

dat = data.frame(x=x, y=as.factor(y))

# Adding huge cost so that misclassification is reduced

svmfit = svm(y~., data = dat, kernel = 'linear', cost = 1e5, scale = F)

summary(svmfit)

plot(svmfit, dat)

# Trying a smaller value of cost to widen the margin

svmfit = svm(y~., data = dat, kernel = 'linear', cost = 1, scale = F)

summary(svmfit)

plot(svmfit, dat)


######################## Support Vector Machine #############################

set.seed(1)

x = matrix(rnorm(200*2), ncol = 2)

x[1:100,] = x[1:100,] + 2

x[101:150,] = x[101:150,] - 2

y = c(rep(1, 150), rep(2, 50))

dat = data.frame(x = x, y = as.factor(y))

plot(x, col = y)

# Data seems linearly inseparable

train = sample(200, 100)

svmfit = svm(y~., data = dat[train,], kernel = "radial", gamma = 1, cost = 1)

plot(svmfit, dat[train,])


# Tuning the model to select best cost and gamma

set.seed(1)

tune.out = tune(svm, y~., data = dat[train,], kernel = 'radial', ranges = list(cost = c(0.1, 1, 10, 100, 1000), 
                gamma = c(0.5, 1, 2, 3, 4)))

summary(tune.out)

bestmod = tune.out$best.model

summary(bestmod)

ypred = predict(bestmod, newdata = dat[-train,])

table(predict = ypred, truth = dat[-train,'y'])

# 10% error rate

###################### ROC Curves ###################################

rocplot = function(pred, truth, ...){
  predob  =  prediction(pred, truth)
  perf  =  performance(predob, "tpr", "fpr") 
  plot(perf, ...)} 

# Note: if the fitted value exceeds zero then the observation is assigned to one class, & 
# if it is less than zero then it is assigned to the other.
# Using decision.values = TRUE when fitting svm() to obtain fitted values and so then predict() 
# function will output the fitted values (not the class) i.e. the distance from boundary

# refit best model now with fitted values
svmfit.opt = svm(y~., data = dat[train,], kernel = "radial", gamma = 2, cost = 1, decision.values = T)

fitted = attributes(predict(svmfit.opt,dat[train,], decision.values = TRUE))$decision.values # grab fitted values

# plot data
rocplot(fitted, dat[train,"y"], main = "Training Data")

# increase γ to produce a more flexible fit (more local behaviour in radial kernel)

svmfit.flex = svm(y~., data = dat[train,], kernel = "radial", gamma = 50, cost = 1, decision.values = T)

fitted = attributes(predict(svmfit.flex,dat[train,],decision.values = T))$decision.values 

rocplot(fitted,dat[train,"y"], add = T, col = "red") 

# Now plot test data ROCR 

fitted = attributes(predict(svmfit.opt, dat[-train,], decision.values = T))$decision.values

rocplot(fitted,dat[-train,"y"], main = "Test Data") 

fitted = attributes(predict(svmfit.flex, dat[-train,], decision.values = T))$decision.values

rocplot(fitted,dat[-train,"y"], add = T, col = "red") 

######################## SVM with Multiple Classes ###############################

# one-versus-one approach

set.seed(1)

x = rbind(x, matrix(rnorm(50*2), ncol = 2))

y = c(y, rep(0,50))

x[y == 0, 2] = x[y == 0, 2] + 2

dat = data.frame(x = x, y = as.factor(y))

plot(x, col = (y + 1)) 

svmfit = svm(y~., data = dat, kernel = "radial", cost = 10, gamma = 1) 

plot(svmfit, dat) 

#################### Application to Gene Expression Data ########################

# We now examine the Khan data set, which consists of a number of tissue
# samples corresponding to four distinct types of small round blue cell tumors.
# For each tissue sample, gene expression measurements are available.
# The data set consists of training data, xtrain and ytrain , and testing data,
# xtest and ytest .

names(Khan)

# check dimensions
dim(Khan$xtrain)

dim(Khan$xtest)

length(Khan$ytrain)

length(Khan$ytest)

# training and test sets consist of 63 and 20 observations respectively.

table(Khan$ytrain)
table(Khan$ytest) 

dat = data.frame(x = Khan$xtrain, y = as.factor(Khan$ytrain)) 

# In this data set, there are a very large number
# of features relative to the number of observations. This suggests that we
# should use a linear kernel, because the additional flexibility that will result
# from using a polynomial or radial kernel is unnecessary

out = svm(y~., data = dat, kernel = "linear", cost = 10) 

summary(out)

table(out$fitted, dat$y) 

# Testing 

dat.te = data.frame(x = Khan$xtest, y = as.factor(Khan$ytest))

pred.te = predict(out, newdata = dat.te)

table(pred.te, dat.te$y)
