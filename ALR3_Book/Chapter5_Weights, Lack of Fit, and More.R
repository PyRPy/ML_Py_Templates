# Applied Linear Regression, Third edition
# Chapter 5
# October 14, 2004; revised January 2011 for alr3 Version 2.0, R only
require(alr3)

head(physics)
m1 <- lm(y ~ x, weights=1/SD^2, data=physics)
s1 <- summary(m1)
print(s1, signif.stars=FALSE)
print(anova(m1), signif.starts=FALSE)

m2 <- update(m1,   ~ . + I(x^2))
s2 <- summary(m2)
print(s2, signif.stars=FALSE)
print(anova(m2), digits=6)

# Fig. 4.1 "../EPS-figs/fig41.eps"
plot(physics$x, physics$y, xlab=expression(paste("x=", s^(-1/2))), 
                              ylab="Cross section,  y") 
abline(m1)
a <- seq(.05, .35, length=50)
lines(a, predict(m2, data.frame(x=a)), lty=2)


#artificial data
x1 <- c(1, 1, 1, 2, 3, 3, 4, 4, 4, 4)
y1 <- c(2.55, 2.75, 2.57, 2.40, 4.19, 4.70, 3.81, 4.87, 2.93, 4.52)
m1 <- lm(y1 ~ x1)
anova(lm(y1 ~ x1 + as.factor(x1), singular.ok=TRUE))
pureErrorAnova(m1)

# short shoot data
a1 <- lm(ybar  ~  Day,  weights=n,  data=longshoots)

# Fig. 5.2 "../EPS-figs/fig42.eps"
plot(ybar ~ Day, longshoots, xlab="Days since dormancy", 
       ylab="Number of stem units")
abline(a1, lty=2)

print(summary(a1), signif.stars=FALSE)
options(digits=7)
anova(a1)

# general F tests,  in the primer,  but not in the book
fuel <- fuel2001 # make a local copy of the data frame
fuel$Dlic <- 1000*fuel$Drivers/fuel$Pop
fuel$Fuel <- 1000*fuel$FuelC/fuel$Pop
fuel$Income <- fuel$Income/1000
fuel$logMiles <- logb(fuel$Miles, 2)
m1 <- lm(Fuel ~ Dlic + Income + logMiles + Tax,  data=fuel)
m2 <- update(m1,   ~ . - Dlic - Income)
m3 <- update(m2,   ~ . - Tax)
anova(m2, m1)
anova(m3, m2, m1) # find more info on this

# confidence regions for the UN data
#UN2 <- read.table("data/UN2.txt", header=TRUE)
m1 <- lm(logFertility  ~  logPPgdp  +  Purban, UN2)
s1 <- summary(m1)

xtx <- (dim(UN2)[1]-1) * var(UN2[, c(1, 2)])
const <- 2 * s1$sigma * qf(.95, 2, dim(UN2)[1]-3)
bhat <- coef(m1)
confidenceEllipse(m1, Scheffe=TRUE)
