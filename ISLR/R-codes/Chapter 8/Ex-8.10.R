library(gbm)
library(ISLR)
library(randomForest)
library(ggplot2)

Hitters = na.omit(Hitters)

Hitters$Salary = log(Hitters$Salary)

train = 1:200

lambda = 10^seq(-10, 10, 1)

errors = rep(NA, length(lambda))

# Bagging 

for(i in 1:length(lambda)){
  
  boost.hitters = gbm(Salary~., data = Hitters[train,], n.trees = 1000, shrinkage = lambda[i], distribution = "gaussian")
  
  preds = predict(boost.hitters, newdata = Hitters[-train,], n.trees = 1000)
  
  errors[i] = mean((preds-Hitters[-train,"Salary"])^2)
  
}

lambda = lambda[!is.na(errors)]

errors = errors[!is.na(errors)]

ggplot(data.frame(lambda, errors), aes(x = lambda, y = errors)) + geom_line() + geom_point(shape=21)

# Constructing model with lowest error associated shrinkage

boost.hitters = gbm(Salary~., data = Hitters[train,], n.trees = 1000, shrinkage = lambda[which.min(errors)], distribution = "gaussian")

pred = predict(boost.hitters, Hitters[-train,], n.trees = 1000)

mean((pred - Hitters[-train,"Salary"])^2)

summary(boost.hitters)

# Relative importance plot in ggplot

df = summary(boost.hitters)

df$var = factor(df$var, levels = df$var[19:1])

ggplot(data.frame(df), aes(x = var, y = rel.inf)) + geom_bar(stat='identity', aes(colour = rel.inf, fill = rel.inf))+coord_flip()+
  labs(x='', y = "Relative Influence")


# Bagging

bag.hitters = randomForest(Salary~., data = Hitters, subset = train, mtry = ncol(Hitters)-1, ntree = 500)

bag.pred = predict(bag.hitters, Hitters[-train,])

mean((bag.pred - Hitters[-train,"Salary"])^2)

# Bagging produces lower error than boosting. Boosted model could be overfitted?