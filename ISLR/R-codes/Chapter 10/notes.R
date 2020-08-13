USArrests

states = row.names(USArrests)

apply(USArrests, 2, mean)

apply(USArrests, 2, var)

prout = prcomp(USArrests, scale. = TRUE)

attributes(prout)

# sd of the principal components
prout$sdev

# mean and sd of the vars prior to scaling
prout$center

prout$scale


# the principal component loadings
prout$rotation

# PC values for the data
prout$x


biplot(prout, scale = 0)

# flipping the signs
prout$rotation = -prout$rotation

prout$x = -prout$x

biplot(prout, scale = 0)

prout.var = prout$sdev^2

# Proportion of variance explained
pve = prout.var/sum(prout.var)

# Scree plot
plot(pve , xlab =" Principal Component ", ylab =" Proportion of Variance Explained ", ylim=c(0,1), type='b')

plot(cumsum(pve), xlab =" Principal Component ", ylab ="Cumulative Proportion of Variance Explained ", ylim=c(0,1), type='b')



##************************ NCI60 data lab ********************************#

library(ISLR)

NCI60
