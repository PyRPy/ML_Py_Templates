# S. Weisberg, Applied Linear Regression, Third edition.
# Chapter 1
# October 14, 2004, revised for version 2.0 of alr3 January 2011

library(alr3)
###########################################################################
# Pearson and Lee data
with(heights,
  plot(Mheight, Dheight, xlim=c(55,75), ylim=c(55,75), pch=20, cex=.3))
head(heights)

sel <- with(heights,
         (57.5 < Mheight) & (Mheight <= 58.5) |
         (62.5 < Mheight) & (Mheight <= 63.5) |
         (67.5 < Mheight) & (Mheight <= 68.5))
library(ggplot2)         
qplot(y=Dheight, x=Mheight, data=heights[sel,])  # to use qplot from 'ggplot2'

with(heights,
     plot(Mheight, Dheight, bty="l", cex=.3, pch=20))
abline(0, 1, lty=2)
abline(lm(Dheight ~ Mheight, data=heights), lty=1)

################################################################################
# forbes
# Table 1.1
forbes

# Fig. 1.3 
oldpar <-par(mfrow=c(1,2), mar=c(4, 3, 1, .5) + .1, mgp=c(2, 1, 0))
plot(Pressure ~ Temp, forbes, xlab="Temperature",ylab="Pressure")
m0 <- lm(Pressure ~ Temp, data=forbes)
abline(m0)
plot(forbes$Temp, residuals(m0), xlab="Temperature", ylab="Residuals",
      bty="l")
abline(h=0, lty=2)
par(oldpar)

# Fig 1.4 forbes01.eps
oldpar <- par(mfrow=c(1, 2), mar=c(4, 3, 1, .5) + .1,
          mgp=c(2, 1, 0))
plot(log10(Pressure) ~ Temp, data=forbes, xlab="Temperature",
     ylab="log(Pressure)")
m0 <- lm(log10(Pressure) ~ Temp, data=forbes)
abline(m0)
plot(forbes$Temp, residuals(m0), xlab="Temperature", ylab="Residuals")
abline(h=0, lty=2)
par(oldpar)

m1 <- lm(log(Pressure) ~ Temp, data=forbes)
m2 <- lm(log10(Pressure) ~ Temp, data=forbes)
anova(m1)
##########################################################################
# wblake.txt
# Fig. 1.5 "../EPS-figs/wblake01.eps"
plot(Length ~ Age, wblake)
abline(lm(Length ~ Age, wblake))
with(wblake, lines(1:8, tapply(Length, Age, mean), lty=2))

###################################################################################
# Fort Collins snowfall
# Fig 1.6 "../EPS-figs/ftcollins1.eps"
head(ftcollinssnow)
plot(Late ~ Early, data=ftcollinssnow,
     xlim=c(0, 60), ylim=c(0, 60))
with(ftcollinssnow, abline(h=mean(Late)))
with(ftcollinssnow, abline(lm(Late ~ Early), lty=2))
m1 <- lm(Late ~ Early, data=ftcollinssnow)

###################################################################################
# turkey data
# Fig. 1.7 
head(turkey)
plot(turkey$A, turkey$Gain, pch=turkey$S, 
       xlab="Amount (percent of diet)", 
       ylab="Weight gain (g)")
legend("bottomright",legend=c("1", "2", "3"),pch=1:3, cex=.7, inset=.02)

###################################################################################
# more heights data
# Fig. 1.10 
plot(Dheight ~ Mheight, data=heights, cex=.1, pch=20)
abline(lm(Dheight ~ Mheight, data=heights), lty=1)
with(heights, 
   lines(lowess(Mheight, Dheight, f=6/10, iter=1), lty=2))

# 1.5 TOOLS FOR LOOKING AT SCATTERPLOTS
# page 13
#anscombe <- read.table("data/anscombe.txt",header=T)
x <- c(anscombe$x1, anscombe$x2, anscombe$x1, anscombe$x1)
y <- c(anscombe$y3, anscombe$y4, anscombe$y1, anscombe$y2)
n <- c("(c)","(d)","(a)","(b)")
g <- factor(rep(n, rep(11,4)),levels=n,ordered=TRUE)
# Fig. 1.9 "../EPS-figs/anscombe1.eps"
require(lattice)
xyplot(y ~ x|g, between=list(x=.32, y=.32), 
       panel = function(x,y) {panel.xyplot(x,y)
                              panel.lmline(x,y)})

###########################################
# Figure 1.11, R
###########################################
fuel2001$Dlic <- 1000*fuel2001$Drivers/fuel2001$Pop
fuel2001$Fuel <- 1000*fuel2001$FuelC/fuel2001$Pop
fuel2001$Income <- fuel2001$Income/1000
fuel2001$logMiles <- log2(fuel2001$Miles)
names(fuel2001)
# Fig. 1.11 "../EPS-figs/fuel02.eps"
pairs(Fuel ~ Tax + Dlic + Income + logMiles,
     data=fuel2001, gap=0.4, cex.labels=1.5)

