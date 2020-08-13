library(MASS)

dataset = Boston

Boston$chas <- factor(Boston$chas, labels = c("N","Y"))
names(Boston)[-1]  # all Bostonthe potential predictors

# extract p-value from model object
lmp <- function (modelobject) {
  f <- summary(modelobject)$fstatistic
  p <- pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) <- NULL
  return(p)
}

results <- combn(names(Boston), 2, 
                 function(x) { lmp(lm(Boston[, x])) }, 
                 simplify = FALSE)
vars <- combn(names(Boston), 2)
names(results) <- paste(vars[1,],vars[2,],sep="~")
results[1:13]  # p-values for response=crim


fit.lm <- lm(crim~., data=Boston)
summary(fit.lm)