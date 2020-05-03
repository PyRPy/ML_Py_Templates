# Applied Linear Regression, Third edition
# Chapter 9 Outliers and Influence
# October 14, 2004; revised January 2011 for alr3 Version 2.0,  R only
require(alr3) 

head(UN2)
tail(UN2)
m1 <- lm(logFertility ~ logPPgdp + Purban, UN2)
center <- coef(m1)
ans <- lm.influence(m1)$coefficients
dim(ans)
dim(UN2)
# Fig. 9.1 
pairs(ans)

scatterplot(ans[, 2], ans[, 3], id.n=6)

# Page 197,  residual for case 12 for Forbes data
m1 <- lm(Lpres  ~  Temp,  data=forbes)
res <- residuals(m1)[12]
sigmahat <- sigmaHat(m1)
lev <- hatvalues(m1)[12]
r12 <- res/(sigmahat*sqrt(1-lev))
data.frame(res=res, sigmahat=sigmahat, lev=lev, r12=r12, 
     rstandard=rstandard(m1)[12], t12=rstudent(m1)[12])

# rat data
# Fig. 9.2 "../EPS-figs/rat1.eps"
head(rat)
pairs(rat)
m1 <- lm(y ~ BodyWt + LiverWt + Dose, rat)
s1<-summary(m1)
print(s1, signif.stars=FALSE)
# Fig. 9.3 "../EPS-figs/rat2.eps"
infIndexPlot(m1)

# page 203
m2 <- update(m1, subset=-3)
s2 <- summary(m2)
print(s2,  signif.stars=FALSE)

# Fig. 9.4 
avPlots(m1, ~ BodyWt + Dose)


# Normal probability plotting
#heights
m1 <- lm(Dheight  ~  Mheight,  heights)
t1 <- lm(Time ~ T1 + T2, transact)

# Fig. 9.5 "../EPS-figs/npp.eps"
oldpar <- par(mfrow=c(1, 2))
qqnorm(residuals(m1), xlab="(a) Heights data", main="")
qqnorm(residuals(t1), xlab="(a) Transaction data", main="")
par(oldpar)
