library(e1071)
library(ggplot2)

## Taken from : 'https://github.com/xw1120/My-Solutions-to-ISLR/blob/master/ch9/ch9.Rmd'

set.seed(3)
t1 = runif(60, 0, 5)
t2 = runif(60, 0, 5)
ind = t2 - t1 > -0.2
x1 = t1[ind]; x2 = t2[ind]
y = rep(1, sum(ind))
t1 = runif(60, 0, 5)
t2 = runif(60, 0, 5)
ind = t2 - t1 < 0.2
x1 = c(x1, t1[ind]); x2 = c(x2, t2[ind])
y = c(y, rep(-1, sum(ind)))
# y = c(rep(1, 30), rep(-1, 30))
plot(0, 0, type="n", xlim=c(0, 5), ylim = c(0, 5), xlab = expression(X[1]), ylab = expression(X[2]))
points(x1[y==1], x2[y==1], pch = 20, cex = 1.5, col = "dodgerblue2")
points(x1[y==-1], x2[y==-1], pch = 20, cex = 1.5, col = "palevioletred3")

## Fitting and predicting

set.seed(1)

tune.out = tune(svm, y~., data = data.frame(x1,x2, y = factor(y)), kernel = 'linear', ranges = list(cost = c(0.1, 1, 10, 100, 1000)))

summary(tune.out)

bestmod = tune.out$best.model

summary(bestmod)

# Increasing cost beyond to 100 and beyond does not decrease the misclassification

## Test data 
# Again taken from : 'https://github.com/xw1120/My-Solutions-to-ISLR/blob/master/ch9/ch9.Rmd' God bless you sir/madam

set.seed(1)
test.x1 = c(runif(20, 0, 3), runif(20, 1.5, 5))
test.x2 = c(runif(20, 1, 5), runif(20, 0, 3.5))
test.y = c(rep(1, 20), rep(-1, 20))
plot(0, 0, type="n", xlim=c(0, 5), ylim = c(0, 5), xlab = expression(X[1]), ylab = expression(X[2]))
points(x1[y==1], x2[y==1], pch = 20, cex = 1.5, col = "dodgerblue2")
points(x1[y==-1], x2[y==-1], pch = 20, cex = 1.5, col = "palevioletred3")
points(test.x1[test.y==1], test.x2[test.y==1], pch = 2, cex = 1.5, col = "dodgerblue2")
points(test.x1[test.y==-1], test.x2[test.y==-1], pch = 4, cex = 1.5, col = "palevioletred3")
# legend("topright", legend = c("Training:class 1", "Training: class -1", "Test: class 1", "Test: class -1"), col = c("dodgerblue2","palevioletred3","dodgerblue2","palevioletred3"), pch = c(20,20,2,4))

## Checking various values of cost

cost = c(0.1, 1, 10, 100, 1000)

dat = data.frame(x1, x2, y = factor(y))

testdat = data.frame(x1 = test.x1, x2 = test.x2, y = factor(test.y))

errors = rep(0, 5)

for(i in 1:5){
  
  svmfit = svm(y~., data = data.frame(x1, x2, y = factor(y)), kernel = 'linear', cost = cost[i])
  
  errors[i] = mean(predict(svmfit, newdata = testdat) == test.y)
  
}

# Costs 0.1, 1 and 1000 have the same accuracy
# Accuracy decreases at 10 and again increases at 100