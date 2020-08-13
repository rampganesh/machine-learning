library(ISLR)
library(tree)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(reshape)

set.seed(1)

train = sample(1:nrow(Carseats), nrow(Carseats)/2)

test = -train

tree.carseats = tree(Sales~., data = Carseats, subset = train)

summary(tree.carseats)

plot(tree.carseats)

text(tree.carseats, pretty=0)


preds = predict(tree.carseats, newdata = Carseats[test,])

mean((Carseats[test,"Sales"]-preds)^2)


# Cross Validation

set.seed(3)

cv.carseats = cv.tree(tree.carseats, FUN = prune.tree)

cv.carseats

plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

# Cross validation shows least deviance on 7 

prune.carseats = prune.tree(tree.carseats, best = 7)

plot(prune.carseats)

text(prune.carseats, pretty=0)

# Testing the pruned tree

preds = predict(prune.carseats, newdata = Carseats[test,])

mean((Carseats[test,"Sales"]-preds)^2)

# Increase in MSE. Trying randomForest.


########################## Random Forest ##################################

bag.carseats = randomForest(Sales~., data = Carseats, subset = train, mtry = 5, importance = T)

preds = predict(bag.carseats, newdata = Carseats[test,])

mean((preds-Carseats[test,"Sales"])^2)

importance(bag.carseats)

varImpPlot(bag.carseats)

# ggplot version

df = importance(bag.carseats)

df=data.frame(df)

colnames(df) = c('IncMSE', 'IncNodePurity')

incmse = ggplot(df, aes(x = IncMSE, y = rownames(df)))+geom_point()+labs(y='')

incnodepurity = ggplot(df, aes(x = IncNodePurity, y = rownames(df)))+geom_point()+labs(y='')

grid.arrange(incmse, incnodepurity, ncol=2)



# Trying permutations of subspace selection and number of trees

mtry = c(10, 5, 3)

ntree = seq(25, 500, 1)

errmat = matrix(rep(NA, length(mtry)*length(ntree)), nrow = length(mtry))

for(i in 1:length(mtry)){
  
  for(j in 1:length(ntree)){
    
    bag.boston = randomForest(Sales~., data = Carseats, subset = train, mtry = mtry[i], ntree = ntree[j])
    
    preds = predict(bag.boston, newdata = Carseats[test,])
    
    errmat[i,j] = mean((preds-Carseats[test,"Sales"])^2)
    
  }
}


errmat = data.frame(errmat, row.names = mtry) 

colnames(errmat) = ntree

errmat_melted = melt(errmat)

errmat_melted$rowid = mtry

ggplot(errmat_melted, aes(variable, value, group=factor(rowid))) + geom_line(aes(color=factor(rowid)))+
  scale_x_discrete(labels = seq(25, 500, 25), breaks = seq(25,500,25))

apply(errmat, 1, min)

which(errmat[1,]==min(errmat[1,]))

# The lowest error appears to be by using the full set of features for each split and 73 trees.

