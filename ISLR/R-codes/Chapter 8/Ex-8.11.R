library(ISLR)
library(gbm)
library(ggplot2)
library(class) 

train = 1:1000

Caravan$Purchase = ifelse(Caravan$Purchase == 'Yes', 1, 0)

boost.caravan = gbm(Purchase~., data = Caravan[train,], shrinkage = 0.01, n.trees = 1000, distribution = 'bernoulli')

summary(boost.caravan)


# Relative importance plot in ggplot

df = summary(boost.caravan)

df$var = factor(df$var, levels = df$var[85:1])

ggplot(data.frame(df), aes(x = var, y = rel.inf)) + geom_bar(stat='identity', aes(colour = rel.inf, fill = rel.inf))+coord_flip()+
  labs(x='', y = "Relative Influence")


# Testing the model

preds = predict(boost.caravan, newdata = Caravan[-train,], n.trees = 1000, type = "response")

preds[preds>0.20]=1

preds[preds<0.20]=0

table(preds, Caravan[-train, "Purchase"])

# 370/4822 = 7.67% error rate


# Logistic regression

glm.fit = glm(Purchase~., data = Caravan[train, ], family = binomial)

preds = predict(glm.fit, newdata = Caravan[-train,], type = "response")

preds[preds>0.20]=1

preds[preds<0.20]=0

table(preds, Caravan[-train, "Purchase"])

# 581/4822 = 12.04% error rate


# KNN classification

train.Y = Purchase[train]
knn.pred = knn(as.matrix(subset(Caravan, select = -Purchase)[train, ]), as.matrix(subset(Caravan, select = -Purchase)[-train, ]), train.Y, k = 1)
table(knn.pred, test.Y)
mean(knn.pred == test.Y)
