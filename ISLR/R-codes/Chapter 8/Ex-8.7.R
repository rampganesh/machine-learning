library(ISLR)
library(randomForest)
library(MASS)
library(ggplot2)

set.seed(1)

train = sample(1:nrow(Boston), nrow(Boston)/2)

test = -train

mtry = c(13, 6, 3)

ntree = seq(25, 500, 1)

errmat = matrix(rep(NA, length(mtry)*length(ntree)), nrow = length(mtry))

for(i in 1:length(mtry)){
  
  for(j in 1:length(ntree)){
    
    bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = mtry[i], ntree = ntree[j])
    
    preds = predict(bag.boston, newdata = Boston[test,])

    errmat[i,j] = mean((preds-Boston[test,"medv"])^2)
    
  }
}


errmat = data.frame(errmat, row.names = mtry) 

colnames(errmat) = ntree

errmat_melted = melt(errmat)

errmat_melted$rowid = mtry

ggplot(errmat_melted, aes(variable, value, group=factor(rowid))) + geom_line(aes(color=factor(rowid)))+
        scale_x_discrete(labels = seq(25, 500, 25), breaks = seq(25,500,25))

# The lowest error appears to be by using 6 random features for each split and around 50 trees.
