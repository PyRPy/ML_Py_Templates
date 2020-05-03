# Applied Linear Regression, Third edition
# Chapter 3 Multiple Regression
# October 14, 2003; revised January 2011 for alr3 Version 2.0, R only

library(alr3)

#UN2 
# Fig. 3.1 
oldpar <- par(mfrow=c(2,2),mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0),
          cex.lab=1.0, pch=".") 
head(UN2)

m1 <- lm(logFertility ~ logPPgdp, UN2)
plot(logFertility ~ logPPgdp, UN2, xlab="(a) log(PPgdp)", ylab = "log(Fertility)")
abline(m1)

print(c("OLS coef for part (a)", round(coef(m1), 4)))

m2 <- lm(logFertility~Purban, UN2)
plot(logFertility ~ Purban, UN2, xlab="(b) Percent urban", ylab = "log(Fertility)")
abline(m2)
print(c("OLS coef for part (b)", round(coef(m2), 4)))

m3 <- lm(Purban ~ logPPgdp, UN2)
plot(Purban ~ logPPgdp, UN2, xlab="(c) log(PPgdp)", ylab="Percent urban")
abline(m3)
print(c("OLS coef for part (c)", round(coef(m3), 4)))

m4 <- lm(resid(m1) ~ resid(m3)) 
plot(resid(m3) ,resid(m1), cex.lab=.7,
    xlab=expression(paste("(d) ", hat(e), " from Purban on log(PPgdp)")),
    ylab=expression(paste(hat(e), " from log(Fertility) on log(PPgdp)"))) 
abline(m4)
print(c("OLS coef for part (d)", round(coef(m4), 4)))
abline(h=0, lty=2)
abline(v=0, lty=2)
par(oldpar)

summary(m1)
summary(m2)


############################################################
#  fuel2001
head(fuel2001)

fuel2001$Dlic <- 1000*fuel2001$Drivers/fuel2001$Pop
fuel2001$Fuel <- 1000*fuel2001$FuelC/fuel2001$Pop
fuel2001$Income <- fuel2001$Income/1000
fuel2001$logMiles <- log2(fuel2001$Miles)

head(fuel2001)

# Fig 3.3 
f <- fuel2001[,c(7, 9, 8, 3, 10)]
pairs(f)
summary(f)

round(cor(f), 4)
round(var(f), 4)

# This figure is not in the text "../EPS-figs/fuel3.eps"
oldpar <- par(mfrow=c(2,2),mai=c(.6,.6,.1,.1),mgp=c(2,1,0))
plot(fuel2001$Dlic, fuel2001$Fuel, xlab="(a) Dlic",ylab="Fuel")

m1 <- lm(Fuel ~ Dlic, data=fuel2001)
abline(m1)

plot(fuel2001$Dlic, fuel2001$Tax, xlab="(b) Dlic", ylab="Tax")

m2 <- lm(Tax ~ Dlic, data=fuel2001)
abline(m2)

plot(fuel2001$Tax, fuel2001$Fuel, xlab="(c) Tax", ylab="Fuel")
abline(lm(Fuel ~ Tax, data=fuel2001))

plot(resid(m1), resid(m2), xlab="(d) Residuals from Tax on Dlic",
                         ylab="Residuals from Fuel on Dlic")
abline(lm(resid(m2)~resid(m1)))
par(oldpar)


# computations using the fuel data
# page 59
f$Intercept <- rep(1, 51)
head(f)

X <- as.matrix(f[, c(6, 1, 3, 4, 5)])
Y <- f$Fuel

xtx <- t(X) %*% X
xtxinv <- solve(xtx)
print(xtxinv, digits=4)

xty <- t(X) %*% Y
print(betahat <- xtxinv %*% xty)

m1 <- lm(formula = Fuel ~ Tax + Dlic + Income + logMiles, data = f)
s1 <- summary(m1)
print(s1, signif.stars=FALSE) # same as betahat

m2 <- update(m1, ~ . - Tax)
anova(m2, m1)

# table 3.5
a1 <- anova(m1)
SSreg <- sum(a1[1:4, 2])
MSreg <- SSreg/4
SSreg
MSreg
a1

m2 <- update(m1, ~Dlic + Income + logMiles + Tax)
anova(m1,m2)
a2 <- anova(m2)

df <- c(3, 1, 46)
SS1 <- a2$"Sum Sq"
SS <- c(sum(SS1[1:3]), SS1[4], SS1[5])
MS <- SS/df
Fval <- MS/MS[3]
atab <- data.frame(df, SS=round(SS, 0), MS=round(MS, 0), Fval)
row.names(atab) <- c("Regression excluding Tax",
                     "Tax after others", "Residual")
round(atab, 2)
m3 <- update(m2, ~ . - Tax)

m4 <- update(m3, ~Dlic + Tax + Income + logMiles)
m5 <- update(m3, ~logMiles + Income + Dlic + Tax)
# Table 3.5
# page 64
print(anova(m4), signif.stars=FALSE)
print(anova(m5), signif.stars=FALSE)

# page 60
m1 <- lm(Fuel ~ Tax + Dlic + Income +logMiles, f)
summary(m1)
