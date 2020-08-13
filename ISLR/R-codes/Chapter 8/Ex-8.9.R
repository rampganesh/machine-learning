library(ISLR)
library(ggplot2)
library(tree)

set.seed(2)

train = sample(1:nrow(OJ), 800)

test = -train

oj.tree = tree(Purchase~., data = OJ[train,], subset = train)

summary(oj.tree)

plot(oj.tree)

text(oj.tree, pretty=0)

preds = predict(oj.tree, newdata = OJ[test,], type = "class")

table(preds, OJ[test,"Purchase"])

# 17.03% error rate

cv.oj = cv.tree(oj.tree, FUN = prune.misclass)

ggplot(data.frame(cv.oj$size, cv.oj$dev), aes(x = cv.oj.size, y = cv.oj.dev)) + geom_point()+geom_line()

# Lowest at 5 nodes

prune.oj = prune.misclass(oj.tree, best = 5)

preds = predict(prune.oj, newdata = OJ[test,], type = "class")

table(preds, OJ[test,"Purchase"])

# 20.37 error rate. The error rate has increased in the pruned tree.