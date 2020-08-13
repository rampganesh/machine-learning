library(ISLR)
library(tree)

attach(Carseats)

High = ifelse(Sales<=8, "No", "Yes")

Carseats = data.frame(Carseats, High)

tree.carseats = tree(High~CompPrice+Income+Advertising+Population+Price+ShelveLoc+Age+Education+Urban+US, data = Carseats)

summary(tree.carseats)

plot(tree.carseats)

text(tree.carseats, pretty=0)

tree.carseats


# Testing the model

set.seed(1)

train = sample(1:nrow(Carseats), 200)

test = -train

tree.carseats = tree(High~CompPrice+Income+Advertising+Population+Price+ShelveLoc+Age+Education+Urban+US, data = Carseats, subset = train)

preds = predict(tree.carseats, Carseats[test,], type = "class")

table(preds, Carseats[test, "High"])

# 46/200 = 77% accuracy

set.seed(3)

cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)

cv.carseats

plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$size, cv.carseats$k, type = "b")

prune.carseats = prune.misclass(tree.carseats, best=16)

plot(prune.carseats)
text(prune.carseats, pretty=0)

tree.pred = predict(prune.carseats, Carseats[test,], type="class")

table(tree.pred, Carseats[test,"High"])


###################### Regression Trees ###############################


set.seed(1)

train = sample(1:nrow(Boston), nrow(Boston)/2)

tree.boston = tree(medv~., data = Boston, subset = train)

summary(tree.boston)

plot(tree.boston)

text(tree.boston, pretty=0)

tree.boston

# Cross Validation

set.seed(1)

train = sample(1:nrow(Boston), nrow(Boston)/2)

test = -train

cv.boston = cv.tree(tree.boston)

cv.boston

plot(cv.boston$size, cv.boston$dev, type = "b")
plot(cv.boston$size, cv.boston$k, type = "b")

prune.boston = prune.tree(tree.boston, best=8)

plot(prune.boston)
text(prune.boston, pretty=0)

# Testing the model

tree.pred = predict(prune.boston, newdata = Boston[test,])

plot(tree.pred, Boston[test,"medv"])

abline(0,1)

mean((tree.pred-Boston[test,"medv"])^2)


########################## Random Forest ###################################

library(randomForest)

set.seed(1)

bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = 13, importance = T)

bag.boston

preds = predict(bag.boston, newdata = Boston[test,])

plot(preds, Boston[test,"medv"])

abline(0,1)

mean((preds-Boston[test,"medv"])^2)


# Reducing the trees

bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = 13, ntree = 25)

preds = predict(bag.boston, newdata = Boston[test,])

plot(preds, Boston[test,"medv"])

abline(0,1)

mean((preds-Boston[test,"medv"])^2)


# Reducing the variables considered at each step

set.seed(1)

bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = 6, importance = T)

preds = predict(bag.boston, newdata = Boston[test,])

plot(preds, Boston[test,"medv"])

abline(0,1)

mean((preds-Boston[test,"medv"])^2)

importance(bag.boston)

# %IncMSE is the OOB error rate. How much does the predictor contribute to the model MSE.
# IncNodePurity indicates the node purity - How much MSE is reduced by splitting on the predictor

varImpPlot(bag.boston)


#################### Boosting ##################################
library(gbm)

set.seed(1)

boost.boston = gbm(medv~., data = Boston[train,], distribution = 'gaussian', n.trees = 5000,
                   interaction.depth = 4)

summary(boost.boston)

plot(boost.boston, i="rm")
plot(boost.boston, i="lstat")

preds = predict(boost.boston, newdata = Boston[test,], n.trees = 5000)

mean((preds-Boston[test,"medv"])^2)

# Using the shrinkage parameter

boost.boston = gbm(medv~., data = Boston[train,], distribution = 'gaussian', n.trees = 5000,
                   interaction.depth = 4, shrinkage = 0.2)

preds = predict(boost.boston, newdata = Boston[test,], n.trees = 5000)

mean((preds-Boston[test,"medv"])^2)
