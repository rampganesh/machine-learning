library(ISLR)
head(Wage)

lm.fit = lm(wage~poly(age, 4), data = Wage)

summary(lm.fit)

attach(Wage)

agelims = range(age)

grid = seq(agelims[1], agelims[2])

y_pred = predict(lm.fit, newdata = list(age=grid), se=T)

pred.bands = cbind(y_pred$fit+ 2*y_pred$se.fit, y_pred$fit - 2*y_pred$se.fit)




par(mfrow = c(1 ,2) , mar = c(4.5 ,4.5 ,1 ,1), oma = c(0 ,0 ,4 ,0))

plot(age , wage , xlim = agelims, cex = .5, col = "darkgrey")

title("Degree -4 Polynomial", outer = T)

lines(grid, y_pred$fit, lwd = 2 , col = "blue")

matlines(grid, pred.bands, lwd = 1, col = "blue", lty =3)


fit2 = lm(wage~poly(age,4,raw=T), data = Wage)

preds2 = predict(fit2, newdata = list(age=grid), se = T)

max(abs(y_pred$fit - preds2$fit))


# CHoosing optimal degree by ANOVA test

fit.1= lm(wage~age, data = Wage)
fit.2= lm(wage~poly(age,2), data = Wage)
fit.3= lm(wage~poly(age,3), data = Wage)
fit.4= lm(wage~poly(age,4), data = Wage)
fit.5= lm(wage~poly(age,5), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5)


coef(summary(fit.5))

# notice the P-values are the same and F values are the square of the t values
(-11.9830341)^2

# ANOVA not only works on orthogonal polynomials but also on other interactions as well

fit.1= lm(wage~education+age, data = Wage)
fit.2= lm(wage~education+poly(age,2), data = Wage)
fit.3= lm(wage~education+poly(age,3), data = Wage)

anova(fit.1, fit.2, fit.3)

# Next we consider the task of predicting whether an individual earns more
# than $250,000 per year. We proceed much as before, except that first we
# create the appropriate response vector, and then apply the glm() function
# using family="binomial" in order to fit a polynomial logistic regression
# model.


fit = glm(I(wage>250)~poly(age,4), data = Wage, family = binomial)

preds = predict(fit, newdata = list(age=grid), se = T)

pfit = exp(preds$fit)/ (1-exp(preds$fit))

se.bands.logit = cbind(preds$fit+ 2*preds$se.fit, preds$fit - 2*preds$se.fit)

se.bands = exp(se.bands.logit)/ (1-exp(se.bands.logit))


# THe probs can also be obtained by using the response option in the predict funciton

preds = predict(fit, newdata = list(age=grid), se = T, type = "response")



plot(age , I(wage>250) , xlim = agelims, type = "n", ylim = c(0,.2))

points(jitter(age), I((wage>250)/5), cex=0.5, pch="|", col = "darkgrey")

title("Degree -4 Polynomial", outer = T)

lines(grid, preds$fit, lwd = 2 , col = "blue")

matlines(grid, se.bands, lwd = 1, col = "blue", lty =3)

# We have drawn the age values corresponding to the observations with wage
# values above 250 as gray marks on the top of the plot, and those with wage
# values below 250 are shown as gray marks on the bottom of the plot. We
# used the jitter() function to jitter the age values a bit so that observations
# with the same age value do not cover each other up. This is often called a
# rug plot.


# Step function

table(cut(age,4))

fit = lm(wage~cut(age,4), data = Wage)
coef(summary(fit))

preds = predict(fit, newdata = list(age=grid), se = T)

se.bands = cbind(preds$fit+ 2*preds$se.fit, preds$fit - 2*preds$se.fit)



plot(age , wage , xlim = agelims, cex = .5, col = "darkgrey")

title("Degree -4 Polynomial", outer = T)

lines(grid, preds$fit, lwd = 2 , col = "blue")

matlines(grid, se.bands, lwd = 1, col = "blue", lty =3)

# Misses a lot of data


################## Splines #######################

library(splines)

fit = lm(wage~bs(age, knots=c(25,40,60)), data = Wage)

preds = predict(fit, newdata = list(age=grid), se = T)

se.bands = cbind(preds$fit+ 2*preds$se.fit, preds$fit - 2*preds$se.fit)


plot(age, wage, col = "grey")

# drawing them knots

segments(25, 0, 25, par('usr')[4])

segments(40, 0, 40, par('usr')[4])

segments(60, 0, 60, par('usr')[4])

lines(grid, preds$fit, lwd = 2)

matlines(grid, se.bands, lwd = 1, col = "red", lty =3)


# Here we have prespecified knots at ages 25, 40, and 60. This produces a
# spline with six basis functions. (Recall that a cubic spline with three knots
# has seven degrees of freedom; these degrees of freedom are used up by an
# intercept, plus six basis functions.) We could also use the df option to
# produce a spline with knots at uniform quantiles of the data.

dim(bs(age, knots = c(25,40,60)))

dim(bs(age, df = 6))

attr(bs(age, df = 6), "knots")


# To fit natural spline

fit2 = lm(wage~ns(age, df= 4), data = Wage)
preds = predict(fit2, newdata = list(age=grid), se = T)
lines(grid, preds$fit, col = "lightblue", lwd = 2)


# To fit a smoothing spline, we use smooth.spline()

plot(age, wage, xlim = agelims, cex=.5, col="darkgrey")

title("Smoothing Spline")

fit = smooth.spline(age, wage, df=16)

fit2 = smooth.spline(age, wage, cv=T)

fit2$df

lines(fit, col = 'red', lwd = 2)

lines(fit2, col = 'blue', lwd = 2)

legend("topright", legend = c("16 DF", "6.8 DF"),
        col = c("red" ,"blue"), lty = 1, lwd = 2, cex = .8)


################# LOESS #####################

plot(age , wage , xlim = agelims , cex =.5 , col =" darkgrey ")
title("Local Regression")
fit = loess(wage~age , span = .2, data = Wage)
fit2 = loess(wage~age, span = .5, data = Wage)
lines(grid, predict(fit, data.frame(age = grid)) ,
      col = "red" , lwd =2)
lines(grid, predict(fit2, data.frame(age = grid)) ,
      col = "blue", lwd = 2)
legend("topright", legend = c("Span = 0.2" ,"Span = 0.5") ,
       col = c("red" ,"blue") , lty = 1 , lwd = 2 , cex = .8)



################## GAM #############################

gam1 = lm(wage~ns(age,5)+ns(year,4)+education, data = Wage)

library(gam)

# The s() function, which is part of the gam library, is used to indicate that
# we would like to use a smoothing spline. We specify that the function of
# year should have 4 degrees of freedom, and that the function of age will
# have 5 degrees of freedom. Since education is qualitative, we leave it as is,
# and it is converted into four dummy variables. We use the gam() function in
# order to fit a GAM using these components.

gam.3 = gam(wage~s(age,5)+s(year,4)+education, data = Wage)

summary(gam.3)

plot(gam.3, se = T, col = "blue")

plot.Gam(gam1, se = T, col = "blue")

# In these plots, the function of year looks rather linear. We can perform a
# series of ANOVA tests in order to determine which of these three models is
# best: a GAM that excludes year (M 1 ), a GAM that uses a linear function
# of year (M 2 ), or a GAM that uses a spline function of year (M 3 ).

gam.m1 = gam(wage~s(age,5)+education, data = Wage)

gam.m2 = gam(wage~year+s(age,5)+education, data = Wage)

anova(gam.m1, gam.m2, gam.3, test = "F")

preds = predict(gam.m2, newdata = Wage)

# We can use the lo() function to fit local regression 

gam.lo = gam(wage~s(year,df=4)+lo(age,span=0.7)+education, data = Wage)

plot.Gam(gam.lo, se = T, col = "green")

# We can use the lo() to create interactions between predictors

gam.lo.i = gam(wage~lo(age,year,span=0.5)+education, data = Wage)

library(akima)

plot(gam.lo.i)

# Fitting the logistic regression model

gam.lr = gam(I(wage>250)~year+s(age,df=5)+education, data = Wage, family = binomial)

plot(gam.lr, se=T, col="green")

table(education,I(wage>250))

# Not graduating High School can only get one so far as per this dataset. Ignore that subset.


gam.lr.s = gam(I(wage>250)~year+s(age,df=5)+education, data = Wage, family = binomial, subset = (education != "1. < HS Grad"))

plot(gam.lr.s, se=T, col="green")