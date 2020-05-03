# Applied Linear Regression, Third edition
# Chapter 8 Regression Diagnostics: Residuals
# need to find other examples for illustration
# October 14, 2004; revised January 2011 for alr3 Version 2.0,  R only
# version$language == "R" for R version$language == NULL for SPlus

require(alr3) 
# Figure 8.1 was done using Arc

# caution data from Cook and Weisberg (1999)
head(caution)

m1 <- lm(y ~ x1 + x2, data=caution)

# Fig. 8.1 "../EPS-figs/caution1.eps"
pairs(y ~ x1 + x2, data=caution)  


# Fig. 8.3 "../EPS-figs/caution2.eps"
plot(fitted(m1), residuals(m1), xlab="Fitted values", ylab="Residuals")
abline(h=0, lty=3, lwd=2)


#fuel2001 <- read.table("data/fuel2001.txt",header=T)

fuel2001$Dlic <- 1000*fuel2001$Drivers/fuel2001$Pop
fuel2001$Fuel <- 1000*fuel2001$FuelC/fuel2001$Pop
fuel2001$Income <- fuel2001$Income/1000

head(fuel2001)

m1 <- lm(formula = Fuel ~ Tax + Dlic + Income + log2(Miles), 
    data = fuel2001)

# Fig. 8.5    
residualPlots(m1, id.n=3, id.method="y")

hatvalues(m1)[c("AK", "DC", "WY")]

# UN data residual curvatures and mmps 
head(UN2)
m1 <- lm(logFertility ~ logPPgdp + Purban, UN2)

# Fig. 8.6 "../EPS-figs/unscatmat.eps"
pairs(~ logPPgdp + Purban + logFertility, UN2)

# Fig. 8.7 "../EPS-figs/unres.eps"
residualPlots(m1)


# Section 8.3 

#snowgeese
m1 <- lm(photo ~ obs1, snowgeese)
head(snowgeese)

# Fig. 8.8 "../EPS-figs/geese1.eps"
pairs(snowgeese)
round(coef(m1), 2)

# page 182
sig2 <- sum(residuals(m1)^2)/length(snowgeese$obs1)
U <- residuals(m1)^2/sig2
m2 <- lm(U ~ snowgeese$obs1)
anova(m2)

#sniffer data
# Fig. 8.9 
pairs(sniffer)

m1 <- lm(Y ~ TankTemp + GasTemp + TankPres + GasPres, sniffer)

# Fig 8.10 s"
op<-par(mfrow=c(2, 2), mar=c(4, 3,0, .5) + .1, mgp=c(2, 1, 0))
plot(predict(m1), residuals(m1), xlab="(a) Yhat", ylab="Residuals")
abline(h=0)
with(sniffer, 
  plot(TankTemp, residuals(m1),xlab="(b) Tank temperature", ylab="Residuals"))
abline(h=0)
with(sniffer, 
  plot(GasPres,residuals(m1),xlab="(c) Gas pressure", ylab="Residuals"))
abline(h=0)

U <- residuals(m1)^2*125/(sum(residuals(m1)^2))
m3 <- update(m1,U ~ .)
plot(predict(m3), residuals(m1), xlab="(d) Linear combination", 
   ylab="Residuals")
abline(h=0)
par(op)


# score tests, using car
library(car)
s1 <- ncvTest(m1, ~ TankTemp, data=sniffer)
s2 <- ncvTest(m1, ~ GasPres, data=sniffer)
s3 <- ncvTest(m1, ~ TankTemp + GasPres, data=sniffer)
s4 <- ncvTest(m1, ~ TankTemp + GasTemp + TankPres + GasPres, data=sniffer)
s5 <- ncvTest(m1)
ans <- data.frame(
      df=c(s1$Df, s2$Df, s3$Df, s4$Df, s5$Df), 
      S=c(s1$ChiSquare, s2$ChiSquare, s3$ChiSquare, s4$ChiSquare, s5$ChiSquare), 
      p=c(s1$p, s2$p, s3$p, s4$p, s5$p))
round(ans,3)


# upper flat creek wc, Western Red Cedar
#ufcwc 
c1 <- lm(Height ~ Dbh, ufcwc)
# Fig. 8.11 "../EPS-figs/ufcwcmmp.eps"
with(ufcwc, mmp(c1, Dbh, label="Diameter, Dbh"))


# UN data, mmp's
# Fig. 8.12 
m1 <- lm(logFertility ~ logPPgdp + Purban,  UN2)

op<-par(mfrow=c(1, 2), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0), cex=0.5)
plot(logFertility ~ logPPgdp, UN2, xlab="(a) log(PPgdp)")
with(UN2, lines(lowess(logPPgdp, logFertility, iter=1, f=2/3)))
plot(predict(m1) ~ logPPgdp, UN2, xlab="(b) log(PPgdp)", ylab="Fitted values")
with(UN2, lines(lowess(logPPgdp, predict(m1), iter=1, f=2/3)) )
par(op)


# Fig. 8.13 "../EPS-figs/unmmp.eps"
mmps(m1,layout=c(2,2))


# Fig. 8.14 "../EPS-figs/unmmp2.eps"
m2 <- update(m1,  ~ . + I(Purban^2))
mmps(m2)

# Fig. 8.15 
op<-par(mfrow=c(1, 2), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0), cex=.5)
mmp(m1, sd=TRUE, label="(a) Fitted values, original model", col=1:2)
mmp(m2, sd=TRUE, label="(b) Fitted values, quadratic term added", col=1:2)
par(op)
