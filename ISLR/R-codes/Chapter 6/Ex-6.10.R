library(leaps)

set.seed(1)
eps = rnorm(1000)
coefs = matrix(rnorm(20000), ncol = 20)
beta = sample(1:10, size = 20, replace = T)

beta[c(2,4,7,10,13)] = 0

Y = coefs %*% beta + eps



train = sample(1:1000, size = 100)

test = -train

x_train = coefs[train,]
x_test = coefs[test,]
y_train = Y[train]
y_test = Y[test]
train_data = data.frame(x_train = x_train, y_train = y_train)
test_data = data.frame(x_test = x_test, y_test = y_test)
colnames(test_data) = colnames(train_data)

regfit.full = regsubsets(y_train~., data = data.frame(x_train = x_train, y_train = y_train), nvmax = 20)

reg.sum = summary(regfit.full)



summary(regfit.full)$rsq

plot(reg.sum$rss, xlab = "Number of variables", ylab = "RSS", type = "l")

plot(reg.sum$adjr2, xlab = "Number of variables", ylab = "Adj Rsq", type = "l")

which.max(reg.sum$adjr2)


predict.regsubsets = function(object , newdata ,id , ...) {
  
  form = as.formula(object$call[[2]])
  
  mat = model.matrix(form, newdata)
  
  coefi = coef(object, id = id)
  
  xvars = names(coefi)
  
  mat[, xvars]%*%coefi
  
}

errors = rep(NA, 20)

for(i in 1:20){
  
  y_pred = predict(regfit.full, newdata = train_data, id=i)
  
  errors[i] = mean((y_pred-y_train)^2)
}


plot(1:20, errors, type = 'l', xlab = "Predictors", ylab = "MSE")

which.min(errors)




test_errors = rep(NA, 20)

for(i in 1:20){
  
  y_pred = predict(regfit.full, newdata = test_data, id=i)
  
  test_errors[i] = mean((y_pred-y_test)^2)
}


plot(1:20, test_errors, type = 'l', xlab = "Predictors", ylab = "Test MSE")

which.min(errors)

coef(regfit.full, id=20)

beta

# Pretty Close. All the coefficients.

beta_error = rep(NA, 20)

for(i in 1:20){
  
  coefs = coef(regfit.full, id = i)[-1]
  
  beta_c = beta[1:i]
  
  beta_error[i] = sqrt(sum((beta_c - coefs)^2))
  
}

plot(1:20, beta_error, type = 'b', xlab = "Predictors", ylab = "Coefficient Errors")

