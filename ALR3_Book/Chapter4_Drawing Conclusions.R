# Applied Linear Regression, Third edition
# Chapter 4
# October 14, 2004; revised January 2011 for alr3 Version 2.0, R only
library(alr3)  

# computations using the fuel data
fuel <- fuel2001  # Not Required in R
fuel$Dlic <- 1000*fuel$Drivers/fuel$Pop
fuel$Fuel <- 1000*fuel$FuelC/fuel$Pop
fuel$Income <- fuel$Income/1000

# use selects the variables of interest including Fuel
m1 <- lm(formula = Fuel ~ Tax + Dlic + Income + log2(Miles), data = fuel)
s1 <- summary(m1)
print(s1, signif.stars=FALSE)
m2 <- update(m1, ~ . -log2(Miles) + log10(fuel$Miles))
summary(m2)

###################################################################
# BGS for BGSgirls only
use <- c(2, 4, 8, 12)
# Fig. 4.1 "../EPS-figs/bgs31.eps"
pairs(BGSgirls[, use])

# print the correlation matrix
print(cor(BGSgirls[, use]), digits=4)

# create linear combinations
BGSgirls$DW9 <- BGSgirls$WT9 - BGSgirls$WT2
BGSgirls$DW18 <- BGSgirls$WT18 - BGSgirls$WT9
BGSgirls$DW218 <- BGSgirls$WT18 - BGSgirls$WT2

m1 <- lm(Soma ~ WT2 + WT9 + WT18 + DW9 + DW18, BGSgirls) 
s1 <- summary(m1)
print(s1, signif.stars=FALSE)

m2 <- lm(Soma ~ WT2 + DW9 + DW18 + WT9 + WT18, BGSgirls)
m3 <- lm(Soma ~ WT2 + WT9 + WT18 + DW9 + DW18, BGSgirls)

# use a car function
compareCoefs(m1, m2, m3, se=FALSE)

s2 <- summary(m2)
print(s2, signif.stars=FALSE)

# Pearson and Lee data
lq <- 61
uq <- 64
outer <- with(heights, lq > Mheight | Mheight > uq)
inner <- !outer
m1 <-lm(Dheight ~ Mheight, heights)
m2 <- update(m1, subset=inner)
m3 <- update(m1, subset=outer)
x <- with(heights, c(Mheight[inner], Mheight[outer], Mheight))
y <- with(heights, c(Dheight[inner], Dheight[outer], Dheight))
g <- rep(c("(c) Inner", "(b) Outer","(a) All"),
       c(length(which(inner)), length(which(outer)), length(outer)))

# Fig. 4.2 
labs <- unique(g)
require(lattice)
xyplot(y~x|factor(g, levels=labs), between=list(y=.32),
   panel=
   function(x,y,...) { panel.xyplot(x, y, pch=20 ,cex=.3)
                       panel.lmline(x, y)}, layout=c(1, 3),
   xlab="Mother's Height", ylab= "Daughter's Height")
unlist(list(m1=summary(m1)$r.squared,
     m2=summary(m2)$r.squared, m3=summary(m3)$r.squared))


# Correlation example
# case 1:  postive bivariate normal
# Fig. 4.3 "../EPS-figs/corr.eps"
cov1 <- matrix(c(1, .7, .7, 1), ncol=2)
cov2 <- matrix(c(1, .00, .00, .08), ncol=2)
cov3 <- matrix(c(1, -.98, -.98, 1), ncol=2)
scov1 <- svd(cov1)$u %*% diag(sqrt(svd(cov1)$d))
scov2 <- svd(cov2)$u %*% diag(sqrt(svd(cov2)$d))
scov3 <- svd(cov3)$u %*% diag(sqrt(svd(cov3)$d))

set.seed(101385)
d <- rnorm(200)
p1 <- matrix(d, ncol=2) %*% t(scov1) %*% diag(c(1, 1))
p2 <- matrix(d, ncol=2) %*% t(scov2) %*% diag(c(1, 3))
p3 <- matrix(d, ncol=2) %*% t(scov3)

#get p4
p4 <- rbind(matrix(2*(.25*rnorm(198) + 1 - 2), ncol=2), matrix(c(4, 4), ncol=2))
# p5
set.seed(101385)
x5 <- 5*(runif(100) - .5)
y5 <- (-x5^2 + 2*x5 + 2*(runif(100) - .5) + 5)/2
p5 <- cbind(x5, y5)
#p6
set.seed(101385)
offset <- rep(c(0, .5, 1, 1.5, 2),c(20, 20, 20, 20, 20))
x6 <- runif(100) + offset
y6 <- -2*x6 + 5*offset
p6 <- cbind(2*(x6 - 2), y6)
d <- rbind(p1, p2, p3, p4, p5, p6)
lab <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
d <- data.frame(x=d[ , 1],y=d[ , 2],g=factor(rep(lab, rep(100, 6)),
      levels=lab[c(5, 6, 3, 4, 1, 2)]))
xyplot(y~x|g, d, scales=list(draw=FALSE), between=list(x=.32, y=.32),
  # strip=strip.default(style=2),
   panel=
   function(x, y, ...) { panel.xyplot(x, y, pch=20 ,cex=.75)
                       panel.lmline(x, y)},layout=c(2, 3),
   xlab=expression(paste("Predictor or ", hat(y))), ylab= "Response")


# sleep data
# page 86, table 4.2
head(sleep1)
dim(sleep1)
sum(is.na(sleep1)) # total NAs

complete.cases(sleep1)

m1 <- lm(log(SWS) ~ log(BodyWt) + log(Life) + log(GP), data=sleep1,
            na.action=na.omit)
m2 <- lm(log(SWS) ~ log(BodyWt)+          log(GP), data=sleep1,
            na.action=na.omit)

use.cases <- complete.cases(sleep1)
m3 <- lm(log(SWS) ~ log(BodyWt) + log(Life) + log(GP), data=sleep1,
            subset = use.cases)

# writing functions
sum.diff <- function(a, b){
   sum <- a + b
   diff <- abs(a - b)
   c(sum=sum, diff=diff)
   }
sum.diff(2, 3)
sum.diff(c(1, 2, 3), c(8, 1, 5))


#############################################################
# Bootstrap
#############################################################
# transactions data
# Fig. 4.4 "../EPS-figs/transact1.eps"
head(transact)
pairs(transact) 
library(boot)
m1 <- lm(Time ~ T1 + T2, data=transact)
betahat <- coef(m1)
# betahat.boot <- boot.case(m1, B=999)
betahat.boot <- boot(m1, B=999) # outdated methods
summary(betahat.boot) # outdated
# bootstrap standard errors
apply(betahat.boot,2,sd)
# bootstrap 95% confidence intervals
cl <- function(x) quantile(x, c(.025, .975))
apply(betahat.boot, 2, cl)

s1 <- summary(m1)
tval <- qt(.975, s1$df[2])
ans.normal <- data.frame(est = s1$coefficients[,1],
                   llim= s1$coefficients[,1] - tval*s1$coefficients[,2],
                   uplim= s1$coefficients[,1] + tval*s1$coefficients[,2])

# the following codes are not working anymore
ans.boot <- t(apply(betahat.boot,2,function(x)(c(mean(x),cl(x)))))

# ratio of beta1/beta2
ratio <- betahat.boot[,2]/betahat.boot[,3]
c(mean(ratio), cl(ratio))


# August 28, 2003
# Rod Pierce's northern pike gillnet data
# catchability data

m0 <- lm(CPUE ~ Density - 1, npdata)
m1 <- update(m0, ~ . + 1)
# Fig. 4.5 "../EPS-figs/npdata1.eps"
plot(CPUE ~ Density, npdata, xlab="Estimated density", ylab="Estimated CPUE",
       xlim=c(0, 50), ylim=c(0, 25))
abline(0, coef(m0))
abline(m1, lty=2)


anova(m0, m1)
s0 <- summary(m0)$coefficients
round(s0[1, 1] + qt(c(.025, .975), 15) * s0[1, 2], 3)

catch.sim <- function(B=999){
 ans <- NULL
 for (i in 1:B) {
   X <- npdata$Density + npdata$SEdens*rnorm(16) 
   Y <- npdata$CPUE + npdata$SECPUE*rnorm(16) 
   m0 <- lm(Y ~ X - 1)
   ans <- c(ans, coef(m0))}
 ans}
 
b0 <- catch.sim(B=999)
c(mean(b0), cl(b0))


   
