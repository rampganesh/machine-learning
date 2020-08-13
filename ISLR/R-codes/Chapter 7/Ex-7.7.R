library(ISLR)
library(splines)
library(gam)

par(mfrow = c(2,4))

attach(Wage)
names(Wage)

gam.fit = gam(wage ~ s(year, df = 7) + lo(age, span = 0.7) + lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
plot.Gam(gam.fit, se = T, col = "green")

gam.m1 = gam(wage ~ lo(age, span = 0.7) + lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
gam.m2 = gam(wage ~ year + lo(age, span = 0.7) + lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
gam.m3 = gam(wage ~ s(year, df = 7) + lo(age, span = 0.7) + lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
anova(gam.m1, gam.m2, gam.m3)

gam.m1 = gam(wage ~ lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
gam.m2 = gam(wage ~ age + lo(age, span = 0.7) + lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
gam.m3 = gam(wage ~ lo(age, span = 0.7) + lo(year, age, span = 0.5) + education + jobclass + maritl + health_ins + race, data = Wage)
anova(gam.m1, gam.m2, gam.m3)


