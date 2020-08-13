library(ISLR)
library(e1071)
library(ggplot2)

head(OJ)

set.seed(1)

train = sample(1:nrow(OJ), size = 800)

############ Fit Linear Kernel SVM ####################

linear.svm = svm(Purchase~., data = OJ[train,], kernel = 'linear', cost = 0.01)

summary(linear.svm)

## training error rate

preds = predict(linear.svm)

table(preds, OJ[train,"Purchase"])

# 16.625% error rate


## Test error rate

preds = predict(linear.svm, newdata = OJ[-train,])

table(preds, OJ[-train,'Purchase'])

# 18.14% error rate


## Tuning to select better cost

cost = c(0.01, 0.05, 0.1, 0.5, 1:10)

svmtune = tune(svm, Purchase ~ ., data = OJ, kernel = 'linear', ranges = list(cost = cost))

svmtune$best.parameters

svmtune$best.performance

## testing the best model training dataset

linear.svm = svm(Purchase~., data = OJ[train,], kernel = 'linear', cost = 1)

preds = predict(linear.svm)

table(preds, OJ[train,"Purchase"])

# 15.875% error rate


## Test error rate

preds = predict(linear.svm, newdata = OJ[-train,])

table(preds, OJ[-train,'Purchase'])

# 19.25% error rate



################# Fit Radial Kernel SVM ####################


radial.svm = svm(Purchase~., data = OJ[train,], kernel = 'radial', cost = 0.01)

summary(radial.svm)

## training error rate

preds = predict(radial.svm)

table(preds, OJ[train,"Purchase"])

# Nearly 50% error rate


## Test error rate

preds = predict(radial.svm, newdata = OJ[-train,])

table(preds, OJ[-train,'Purchase'])

# 41% error rate. 


## Tuning to select better cost

cost = c(0.01, 0.05, 0.1, 0.5, 1:10)

svmtune = tune(svm, Purchase ~ ., data = OJ, kernel = 'radial', ranges = list(cost = cost))

svmtune$best.parameters

svmtune$best.performance

## testing the best model training dataset

radial.svm = svm(Purchase~., data = OJ[train,], kernel = 'radial', cost = 0.5)

preds = predict(radial.svm)

table(preds, OJ[train,"Purchase"])

# 14.75% error rate


## Test error rate

preds = predict(radial.svm, newdata = OJ[-train,])

table(preds, OJ[-train,'Purchase'])

# 16.67% error rate



################### Fit Polynomial Kernel SVM ########################


poly.svm = svm(Purchase~., data = OJ[train,], kernel = 'polynomial', cost = 0.01, degree = 2)

summary(poly.svm)

## training error rate

preds = predict(poly.svm)

table(preds, OJ[train,"Purchase"])

# Nearly 50% error rate. Same as radial svm's.


## Test error rate

preds = predict(poly.svm, newdata = OJ[-train,])

table(preds, OJ[-train,'Purchase'])

# 41% error rate. Again, same as radial kernel.


## Tuning to select better cost

cost = c(0.01, 0.05, 0.1, 0.5, 1:10)

svmtune = tune(svm, Purchase ~ ., data = OJ, kernel = 'polynomial', ranges = list(cost = cost), degree = 2)

svmtune$best.parameters

svmtune$best.performance

## testing the best model training dataset

poly.svm = svm(Purchase~., data = OJ[train,], kernel = 'polynomial', cost = 8, degree = 2)

preds = predict(poly.svm)

table(preds, OJ[train,"Purchase"])

# 14.5% error rate


## Test error rate

preds = predict(poly.svm, newdata = OJ[-train,])

table(preds, OJ[-train,'Purchase'])

# 18.89% error rate



# The radial kernel seems to give the best result on the data. The polynomial kernel performs slightly better than the
# linear kernel.

# It is worth noting, in case of radial, increasing the cost 5000 % from 0.01 to 0.5 reduced the test/train error by half.
# Similarly, for polyomial, increasing the cost 8000% from 0.01 to 8 reduced errors by half.