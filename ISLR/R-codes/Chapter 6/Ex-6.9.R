library(ISLR)
library(glmnet)
library(pls)
options(scipen=999)


College$Private = factor(College$Private, labels = c(1,0), levels = c('Yes','No')) 

set.seed(1)

train = sample(1:nrow(College), size = nrow(College)/2, )

test = (-train)

X = College[,-2]

y = College[,2]

######### Linear Model ####################

fit.lm = lm("Apps~.", College[train,])

y_pred = predict(fit.lm, newdata = College[test,])

lm.err = mean((y_pred - College[test,"Apps"])^2)

########## Ridge Regression #################

set.seed(1)

grid = 10^seq(10, -2, length.out = 100)

X = model.matrix(Apps~., College)[,-1]

regressor = glmnet(X[train,], y[train], lambda = grid, alpha = 0)

cv.out = cv.glmnet(X[train,], y[train], alpha = 0)

plot(cv.out)

bestlam = cv.out$lambda.min

bestlam

ridge.pred = predict(regressor, s = bestlam, newx=X[test,])

ridge.err = mean((ridge.pred-y[test])^2)

########### Lasso Regression ###############

set.seed(1)

regressor = glmnet(X[train,], y[train], lambda = grid, alpha = 1)

cv.out = cv.glmnet(X[train,], y[train], alpha = 1)

plot(cv.out)

bestlam = cv.out$lambda.min

bestlam

lasso.pred = predict(regressor, s = bestlam, newx=X[test,])

lasso.err = mean((lasso.pred-y[test])^2)

sum(predict(regressor, s = bestlam, newx=X[test,], type = "coefficients") == 0)


################### PCR ######################

set.seed(1)

pcr.fit = pcr(Apps~., data = College, scale = T, validation = 'CV')

summary(pcr.fit)

validationplot(pcr.fit, val.type = "MSEP")

# variance explained is max by 10. we'll consider 11

y_pred = predict(pcr.fit, X[test,], ncomp = 11)

pcr.err = mean((y_pred - y[test])^2)

set.seed(1)

pcr.fit = pcr(Apps~., data = College, scale = T, ncomp = 11)

summary(pcr.fit)


################# PLS Reg #########################

set.seed(1)

pls.fit = plsr(Apps~., data = College, scale = T, validation = 'CV', subset = train)

summary(pls.fit)

validationplot(pls.fit, val.type = "MSEP")



pls.pred = predict(pls.fit, X[test,], ncomp = 10)

pls.err = mean((pls.pred - y[test])^2)


errors = c(lm.err, ridge.err, lasso.err, pcr.err, pls.err)
names(errors) = c('Linear','Ridge','Lasso','PCR','PLS')
barplot(errors)
