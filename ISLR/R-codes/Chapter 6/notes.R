library(ISLR)
library(leaps)
Hitters

sum(is.na(Hitters))

Hitters = na.omit(Hitters)

library(leaps)

regfit.full= regsubsets(Salary~., Hitters)

summary(regfit.full)

# the variables hits and crbi seem to be included in all the iterations

regfit.full= regsubsets(Salary~., data=Hitters, nvmax =19)

reg.sum = summary(regfit.full)

summary(regfit.full)$rsq

plot(reg.sum$rss, xlab = "Number of variables", ylab = "RSS", type = "l")

plot(reg.sum$adjr2, xlab = "Number of variables", ylab = "Adj Rsq", type = "l")

which.max(reg.sum$adjr2)

points(11, reg.sum$adjr2[11], col = "red", cex = 2, pch = 20)

plot(regfit.full, scale = "adjr2")




############################### CV and validation set approach ##################################

set.seed(1)

train = sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)

test = !train

regfit.best = regsubsets(Salary~., Hitters[train,], nvmax = 19)

test.mat = model.matrix(Salary~., data = Hitters[test,])


val.errors = rep(NA,19)

for(i in 1:19){
  
    coefi = coef(regfit.best,id = i)
    
    # Matrix multiplication
    
    pred = test.mat[, names(coefi)]%*%coefi
    
    val.errors[i] = mean((Hitters$Salary[test] - pred)^2)
}

which.min(val.errors)

predict.regsubsets = function(object , newdata ,id , ...) {
  
   form = as.formula(object$call[[2]])
   
   mat = model.matrix(form, newdata)
   
   coefi = coef(object, id = id)
   
   xvars = names(coefi)
   
   mat[, xvars]%*%coefi
   
}

regfit.best = regsubsets(Salary~., data = Hitters, nvmax = 19)

coef(regfit.best, 10)



############################ Cross Validation ################################

# We are performing Leave One Out CV here

k = 10

set.seed(1)

folds = sample(1:k, nrow(Hitters), replace = T)

cv.errors = matrix(NA ,k ,19, dimnames = list(NULL , paste(1:19)))

for (j in 1:k){
  
  best.fit = regsubsets(Salary~., data = Hitters[folds != j,],
                          nvmax = 19)
  for (i in 1:19){
    
    pred = predict(best.fit, Hitters[folds == j,], id = i)
    
    cv.errors[j,i] = mean((Hitters$Salary[folds == j] - pred)^2)
    
  }
}

mean.cv.errors = apply(cv.errors, 2, mean)

which.min(mean.cv.errors)

plot(mean.cv.errors, type = 'b')


regfit.best = regsubsets(Salary~., data = Hitters, nvmax = 19)

coef(regfit.best, 11)




########################### Ridge Regression #############################

x = model.matrix( Salary~., Hitters)[,-1]

y = Hitters$Salary


library(glmnet)

grid = 10^seq(10,-2, length = 100)

ridge.mod = glmnet(x,y, alpha = 0, lambda = grid)

dim(coef(ridge.mod))

ridge.mod$lambda[50]

coef(ridge.mod)[,50]

sqrt(sum(coef(ridge.mod )[-1, 50]^2))



ridge.mod$lambda[60]

coef(ridge.mod)[,60]

sqrt(sum(coef(ridge.mod )[-1, 60]^2))


predict(ridge.mod, s = 50, type = "coefficients")[1:20 ,]



set.seed (1)
train = sample(1: nrow(x), nrow(x)/2)
test = (- train )
y.test = y[ test]


ridge.mod =glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12)

ridge.pred = predict(ridge.mod, s = 4, newx=x[test,])

mean((ridge.pred-y.test)^2)

mean((mean(y[train ])-y.test)^2)



ridge.pred = predict(ridge.mod, s = 1e10, newx=x[test,])

mean((ridge.pred-y.test)^2)


ridge.pred = predict(ridge.mod, s = 0, newx=x[test,])

mean((ridge.pred-y.test)^2)


lm("y~x", subset = train)

predict(ridge.mod, s = 0, exact = T, type = "coefficients")[1:20 ,]


# Selecting best lambda using cross validation

set.seed(1)

cv.out = cv.glmnet(x[train,], y[train], alpha = 0)

plot(cv.out)

bestlam = cv.out$lambda.min

bestlam

ridge.pred = predict(ridge.mod, s = bestlam, newx=x[test,])

mean((ridge.pred-y.test)^2)


out = glmnet(x, y, alpha = 0)

predict(out, type = "coefficients", s = bestlam)[1:20,]



#################### Lasso Regression ##########################

lasso.mod = glmnet(x[train,], y[train], alpha = 1, lambda = grid)

plot(lasso.mod)


set.seed(1)

cv.out = cv.glmnet(x[train,], y[train], alpha = 1)

plot(cv.out)

bestlam = cv.out$lambda.min

bestlam

lasso.pred = predict(lasso.mod, s = bestlam, newx=x[test,])

mean((lasso.pred-y.test)^2)




out = glmnet(x, y, alpha = 1, lambda = grid)

predict(out, type = "coefficients", s = bestlam)[1:20,]


# Note that Lasso here does feature selection as can be seen clearly from some coeffs
# being assigned 0





################## Principal Component Regression #######################


library(pls)

set.seed(2)

pcr.fit = pcr(Salary~., data = Hitters, scale = T, validation = 'CV')

summary(pcr.fit)

validationplot(pcr.fit, val.type = "MSEP")




set.seed(1)

pcr.fit = pcr(Salary~., data = Hitters, scale = T, validation = 'CV', subset = train)

summary(pcr.fit)

validationplot(pcr.fit, val.type = "MSEP")

pcr.pred = predict(pcr.fit, x[test,], ncomp = 7)

mean((pcr.pred - y[test])^2)




pcr.fit = pcr(Salary~., data = Hitters, scale = T, ncomp = 7)

summary(pcr.fit)




################### Partial Least squares Regression #########################


set.seed(1)

pls.fit = plsr(Salary~., data = Hitters, scale = T, validation = 'CV', subset = train)

summary(pls.fit)

validationplot(pls.fit, val.type = "MSEP")



pls.pred = predict(pls.fit, x[test,], ncomp = 2)

mean((pls.pred - y[test])^2)




pls.fit = plsr(Salary~., data = Hitters, scale = T, ncomp = 2)

summary(pls.fit)
