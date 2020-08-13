library(ISLR)
library(leaps)
library(MASS)
Boston

sum(is.na(Boston))

Boston = na.omit(Boston)

library(leaps)

regfit.full= regsubsets(crim~., Boston)

summary(regfit.full)

# the variables hits and crbi seem to be included in all the iterations

regfit.full= regsubsets(crim~., data=Boston, nvmax =13)

reg.sum = summary(regfit.full)

summary(regfit.full)$rsq

plot(reg.sum$rss, xlab = "Number of variables", ylab = "RSS", type = "l")

plot(reg.sum$adjr2, xlab = "Number of variables", ylab = "Adj Rsq", type = "l")

# which.max(reg.sum$adjr2)

points(which.max(reg.sum$adjr2), reg.sum$adjr2[which.max(reg.sum$adjr2)], col = "red", cex = 2, pch = 20)

plot(regfit.full, scale = "adjr2")




############################### CV and validation set approach ##################################

set.seed(1)

train = sample(c(TRUE, FALSE), nrow(Boston), rep = TRUE)

test = !train

regfit.best = regsubsets(crim~., Boston[train,], nvmax = 13)

test.mat = model.matrix(crim~., data = Boston[test,])


val.errors = rep(NA,13)

for(i in 1:13){
  
  coefi = coef(regfit.best,id = i)
  
  # Matrix multiplication
  
  pred = test.mat[, names(coefi)]%*%coefi
  
  val.errors[i] = mean((Boston$crim[test] - pred)^2)
}

which.min(val.errors)

predict.regsubsets = function(object , newdata ,id , ...) {
  
  form = as.formula(object$call[[2]])
  
  mat = model.matrix(form, newdata)
  
  coefi = coef(object, id = id)
  
  xvars = names(coefi)
  
  mat[, xvars]%*%coefi
  
}

regfit.best = regsubsets(crim~., data = Boston, nvmax = 13)

coef(regfit.best, 2)

y_pred = predict(regfit.best, Boston[test,], 2)

best_subset_err = mean((y_pred - Boston$crim[test])^2)


############################ Cross Validation ################################

# We are performing Leave One Out CV here

k = 10

set.seed(1)

folds = sample(1:k, nrow(Boston), replace = T)

cv.errors = matrix(NA ,k ,13, dimnames = list(NULL , paste(1:13)))

for (j in 1:k){
  
  best.fit = regsubsets(crim~., data = Boston[folds != j,],
                        nvmax = 13)
  for (i in 1:13){
    
    pred = predict(best.fit, Boston[folds == j,], id = i)
    
    cv.errors[j,i] = mean((Boston$crim[folds == j] - pred)^2)
    
  }
}

mean.cv.errors = apply(cv.errors, 2, mean)

which.min(mean.cv.errors)

plot(mean.cv.errors, type = 'b')


regfit.best = regsubsets(crim~., data = Boston, nvmax = 13)

y_pred = predict(regfit.best, Boston[test,], 12)

cv_val_err = mean((y_pred - Boston$crim[test])^2)

########################### Ridge Regression #############################

x = model.matrix( crim~., Boston)[,-1]

y = Boston$crim


library(glmnet)

grid = 10^seq(10,-2, length = 100)

ridge.mod = glmnet(x,y, alpha = 0, lambda = grid)

dim(coef(ridge.mod))


set.seed(1)
train = sample(1: nrow(x), nrow(x)/2)
test = (- train )
y.test = y[ test]



# Selecting best lambda using cross validation

set.seed(1)

cv.out = cv.glmnet(x[train,], y[train], alpha = 0)

plot(cv.out)

bestlam = cv.out$lambda.min

bestlam

ridge.pred = predict(ridge.mod, s = bestlam, newx=x[test,])

ridge_err = mean((ridge.pred-y.test)^2)


#################### Lasso Regression ##########################

lasso.mod = glmnet(x[train,], y[train], alpha = 1, lambda = grid)

plot(lasso.mod)


set.seed(1)

cv.out = cv.glmnet(x[train,], y[train], alpha = 1)

plot(cv.out)

bestlam = cv.out$lambda.min

bestlam

lasso.pred = predict(lasso.mod, s = bestlam, newx=x[test,])

lasso_err = mean((lasso.pred-y.test)^2)

# Note that Lasso here does feature selection as can be seen clearly from some coeffs
# being assigned 0

################## Principal Component Regression #######################


library(pls)

set.seed(1)

pcr.fit = pcr(crim~., data = Boston, scale = T, validation = 'CV')

summary(pcr.fit)

# 97 % of the variance is explained by the 10th PC

validationplot(pcr.fit, val.type = "MSEP")



set.seed(1)

pcr.fit = pcr(crim~., data = Boston, scale = T, validation = 'CV', subset = train)

summary(pcr.fit)

validationplot(pcr.fit, val.type = "MSEP")

pcr.pred = predict(pcr.fit, x[test,], ncomp = 10)

pcr_err = mean((pcr.pred - y[test])^2)


################### Partial Least squares Regression #########################


set.seed(1)

pls.fit = plsr(crim~., data = Boston, scale = T, validation = 'CV', subset = train)

summary(pls.fit)

# 10 components seem like a good fit. will verify.

validationplot(pls.fit, val.type = "MSEP")


ncs = 9:12

errors = rep(NA, 4)

for(i in 1:4){
  
  pls.pred = predict(pls.fit, x[test,], ncomp = ncs[i])
  
  errors[i] = mean((pls.pred - y[test])^2)
  
}

which.min(errors)

# 10

pls.pred = predict(pls.fit, x[test,], ncomp = 10)

pl_err = mean((pls.pred - y[test])^2)


errors = c(best_subset_err, cv_val_err, ridge_err, lasso_err, pcr_err, pl_err)

barplot(errors, xlab = "Model Selection", ylab = "Test MSE", names.arg = 
          c("Best sub", "CV", "Ridge", "Lasso", "PCR", "PLS"))


# Ridge performs the best among all of the aproaches while Lasso does come a close second.
# Either Lasso or Ridge can be chosen. 
# But we did observe that Lasso does feature selection. So, it might be preferable.
