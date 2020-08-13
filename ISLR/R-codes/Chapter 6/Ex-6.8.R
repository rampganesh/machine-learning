library(glmnet)
library(leaps)

set.seed(1)

X = rnorm(n = 100)

eps = rnorm(n = 100)

Y = 7 + 9*X + 11 * X^2 - 2*X^3 + eps

mods = regsubsets(Y~poly(X,10,raw=T), data=data.frame(Y,X), nvmax=10)

mods.sum = summary(mods)

which.min(mods.sum$cp)

which.min(mods.sum$bic)

which.max(mods.sum$adjr2)


plot(mods.sum$rss, xlab = "Number of variables", ylab = "RSS", type = "l")

plot(mods.sum$adjr2, xlab = "Number of variables", ylab = "Adj Rsq", type = "l")






########## Forward selection & Backward selection ###########



mods.fwd = regsubsets(Y~poly(X,10,raw=T), data=data.frame(Y,X), nvmax=10, method = "forward")

mods.bwd = regsubsets(Y~poly(X,10,raw=T), data=data.frame(Y,X), nvmax=10, method = "backward")

# Forward

modsf.sum = summary(mods.fwd)

which.min(modsf.sum$cp)

which.min(modsf.sum$bic)

which.max(modsf.sum$adjr2)

# Backward

modsb.sum = summary(mods.bwd)

which.min(modsb.sum$cp)

which.min(modsb.sum$bic)

which.max(modsb.sum$adjr2)


###################### Lasso #############################

mod.mat = model.matrix(Y~poly(X,10,raw=T))[,-1]

lasso.mod <- cv.glmnet(mod.mat, Y, alpha=1)

bestlam <- lasso.mod$lambda.min


plot(lasso.mod)

predict(lasso.mod, s=bestlam, type="coefficients")





################# Best Subsets ###################


Y = 7 + 2.3*X^7 + eps



mods = regsubsets(Y~poly(X,10,raw=T), data=data.frame(Y,X), nvmax=10)

mods.sum = summary(mods)

which.min(mods.sum$cp)

which.min(mods.sum$bic)

which.max(mods.sum$adjr2)


plot(mods.sum$rss, xlab = "Number of variables", ylab = "RSS", type = "l")

plot(mods.sum$adjr2, xlab = "Number of variables", ylab = "Adj Rsq", type = "l")

plot(mods.sum$cp, xlab = "Number of variables", ylab = "CP", type = "l")

plot(mods.sum$bic, xlab = "Number of variables", ylab = "BIC", type = "l")



############## Lasso ###################


mod.mat = model.matrix(Y~poly(X,10,raw=T))[,-1]

lasso.mod <- cv.glmnet(mod.mat, Y, alpha=1)

bestlam <- lasso.mod$lambda.min


plot(lasso.mod)

predict(lasso.mod, s=bestlam, type="coefficients")

# Lasso selects the correct 
