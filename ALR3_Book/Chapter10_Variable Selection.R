# Chapter 10 variable selection
# October 14, 2004; revised January 2011 for alr3 Version 2.0,  R only
require(alr3)

# simulation example
set.seed(1013185)
case1 <- data.frame(x1=rnorm(100), x2=rnorm(100),
                    x3=rnorm(100), x4=rnorm(100))
e <- rnorm(100)
case1$y <- 1 + case1$x1 + case1$x2 + e
m1 <- lm(y ~ x1 + x2 + x3 + x4, data=case1)


X <- as.matrix(case1[, -5]) # change from data.frame to a matrix, drop y
Var2 <- matrix(c(1,   0, .95,   0,
                 0,   1,   0,-.95,
               .95,   0,   1,   0,
                 0,-.95,   0,   1), ncol=4)
s1 <- chol(Var2)  # cholesky factor of Var2
X <- X %*% s1
dimnames(X)[[2]] <- paste("x", 1:4, sep="")
case2 <- data.frame(X)
case2$y <- 1 + case2$x1 + case2$x2 + e
m2 <- lm(y ~ x1 + x2 + x3 + x4, data=case2)

sum1 <- summary(m1)
sum2 <- summary(m2)
varb1 <- vcov(m1)
varb2 <- vcov(m2)
sum1
100*varb1
sum2
100*varb2

# Case 2, but with (much) larger sample size
m <- 1100
X2 <- matrix(rnorm(4*m), ncol=4) %*% s1
dimnames(X2)[[2]] <- paste("x",1:4,sep="")
beta <- c(1, 1, 0, 0)
e <- rnorm(m)
Y2 <- 1 + X2 %*% beta + e
m3 <- lm(Y2 ~ X2) 
sum3 <- summary(m3)
varb3 <- sum3$sigma^2 * sum3$cov.unscaled
sum3
m*varb3

# Case 2, but fitting only with the variables x1 and x4

m2a <- update(m2, ~ . - x2 - x3)
sum2a <- summary(m2a)
sum2a

m2b <- update(m2, ~ . - x1 - x2)
sum2b <- summary(m2b)
sum2b


# variance inflation factors
vif(m1)
vif(m2)


# page 222 - detailed explation for the procedures
# highway data
highway$Sigs1 <- with(highway, (Sigs*Len+1)/Len)
head(highway)
cols <- c(17,15,13,14,16,7,10,3,4,6,9,11)
m1 <- lm(log2(Rate) ~ log2(Len) + log2(ADT) + log2(Trks) +
                      log2(Sigs1) + Slim + Shld +
                      Lane + Acpt + Itg + Lwid + Hwy, highway)

summary(m1)

# step forward AIC
m0 <- lm(log2(Rate) ~ log2(Len), data=highway)
ansf1 <- step(m0, scope=list(lower= ~ log2(Len),
         upper=~ log2(Len) + log2(ADT) + log2(Trks) + log2(Sigs1) +
                Slim + Shld + Lane + Acpt + Itg + Lwid + Hwy),
                direction="forward", data=highway)

ansf2 <- step(m1, scope=list(lower= ~ log2(Len),
         upper= ~ log2(Len) + log2(ADT) + log2(Trks) + log2(Sigs1) +
                Slim + Shld + Lane + Acpt + Itg + Lwid + Hwy),
                direction="backward", data=highway, scale=sigmaHat(m1)^2)

 m2 <- update(m1, . ~ log2(Len) + Slim + Acpt + log2(Trks) + Shld)
 n <- dim(highway)[1]  # sample size
 anova(m1, m2)
 deviance(m2)
 
 extractAIC(m2)
 extractAIC(m2, k=log(n))
 
 deviance(m2)/(summary(m1)$sigma^2) - n + 2*(n - m2$df.residual)
 sum( (residuals(m2)/(1 - hatvalues(m2)))^2 )
 deviance(m1)
 
 extractAIC(m1)
 extractAIC(m1, k=log(n))
 deviance(m1)/(summary(m1)$sigma^2) - n + 2*(n - m1$df.residual)
 sum( (residuals(m1)/(1-hatvalues(m1)))^2 )
 m3 <- update(m1, ~ log2(Len) + Slim + log2(Trks) + Hwy + log2(Sigs1))
 summary(m3)


# Freedman simulation
sim.ans <- function(s1){
 r2 <- s1$r.squared
 g1 <- which(abs(s1$coe[-1, 3]) > sqrt(2))
 g2 <- which(abs(s1$coe[-1, 3]) > 2)
 f <- s1$fstatistic
 list(r2=r2, pval=1 - pf(f[1], f[2], f[3]),  
      nsqrt2=length(g1), n2=length(g2), g1=g1, g2=g2)
 }

set.seed(3241944)
X <- matrix(rnorm(100*50), ncol=50)
Y <- rnorm(100)
s1 <- summary(lm(Y ~ X))
a1 <- sim.ans(s1)
a1

s2 <- summary(lm(Y~X[, a1$g1]))
sim.ans(s2)

s3 <- summary(lm(Y~X[, a1$g2]))
sim.ans(s3)


# page 
#Windmills
wm4$bin1 <- factor(wm4$bin1)
head(wm4)

m1 <- lm(CSpd ~ Spd1, wm4)
m2 <- update(m1, ~ Spd1*bin1)
m3 <- update(m1, ~ Spd1 + cos(Dir1) + sin(Dir1) + Spd1:(cos(Dir1) + sin(Dir1)))
m4 <- update(m1, ~ Spd1 + Spd1Lag1)
m5 <- update(m1, ~ Spd1 + Spd2 + Spd3 + Spd4)
m6 <- update(m5, ~ . + Spd1Lag1 + Spd2Lag1 + Spd3Lag1 + Spd4Lag1)

# getAIC and getBIC are in ALR3.S

fit.summary <- function(m){
      n <- m$df.residual + m$rank
      aic <- extractAIC(m)
      bic <- extractAIC(m,k=log(n)) 
      data.frame(edf=aic[1],AIC=aic[2],BIC=bic[2],
                 PRESS= sum( (residuals(m)/(1-hatvalues(m)))^2 ))} 
ans <- fit.summary(m1)
ans <- rbind(ans,fit.summary(m2))
ans <- rbind(ans,fit.summary(m3))
ans <- rbind(ans,fit.summary(m4))
ans <- rbind(ans,fit.summary(m5))
ans <- rbind(ans,fit.summary(m6))
row.names(ans) <- paste("Model",1:6)
ans


# windmill data simulation, section 10.4.2
# Revised December 26, 2003

# I've labelled the four sites
# S1 = (105, 55)
# S2 = (105, 56)
# S3 = (106, 55)
# S4 = (106, 56)

#########
##  wm5.txt
#########
# The file wm5 is NOT part of the alr3 library; you need to download it 
# separately from www.stat.umn.edu/alr.

# The following will work in R, but in Splus you need to download the file

d <- read.table("wm5.txt", header=TRUE)
attach(d)   # to avoid writing d$Dir1
sel <- Year == 2002 # selector for 2002 data

# define bins
# bb gives the bin boundaries with 16 bins
bb <- (0:16)*22.5

# bb1 gives the bin boundaries with just 1 bin, so 
# direction is effectively ignored
bb1 <- c(0, 360)

# regression functions
# function for the within-bin regression

reg1 <- function(x, y, dir, bb, bin,sel){
# select points in the bin
     sel1 <- bb[bin] < dir  & bb[bin+1] >= dir
     m1 <- lm(y ~ x, subset=sel & sel1) # within bin for 2002
# compute the mean of x for the bin and # of pts in the bin
     n <- length(x[sel1])
     xmean <- mean(x[sel1])
# predict at xmean and get standard error
     p <- predict(m1,data.frame(x=xmean),se.fit=TRUE)
# return the estimated mean, its se and n
     c(p$fit, sqrt(p$residual.scale^2/length(x[sel1]) + p$se.fit^2), n)
}

# function to compute for all the bins
reg2 <- function(x, y, dir, bb=bb, sel=Year==2002, compare=FALSE){
     nbins <- length(bb) - 1
     ans <- NULL
     for (i in 1:nbins){ans <- rbind(ans, reg1(x, y, dir, bb, i, sel))}
     ans <- data.frame(ans)
     names(ans) <- c("EstSpeed", "SE", "n")
# get combined answer
     sumn <- sum(ans$n)
     ave <- sum (ans$EstSpeed*ans$n) / sumn
     se <- sqrt(sum( (ans$n/sumn)^2 * ans$SE^2))
     ans <- rbind(ans,c(ave, se, sumn))
     row.names(ans)[nbins+1] <- "Combined"
     if (compare==TRUE){
       ans <- rbind(ans,c(mean(y), sd(y)/sqrt(sumn), sumn))
       row.names(ans)[nbins+2] <- "Actual"}
     ans
}

ns <- length(which(sel))  # number of observations in 2002
N <- length(Spd1)         # number of observations in the data file.

one.sim <- function(y=Spd1, x1=Spd2, x2=Spd3, x3=Spd4,
                           d1=Dir1, d2=Dir2, d3=Dir3) {
 ns <- length(which(sel))  # number of observations in 2002
 sel1 <- rep(FALSE, N)
 sel1[sample(1:N, ns,  replace=FALSE)] <- TRUE  # create a psuedo-year
 a161 <- reg2(x1, y, d1, bb, sel1)[17, c(1, 2)]   
 a162 <- reg2(x2, y, d2, bb, sel1)[17, c(1, 2)] 
 a163 <- reg2(x3, y, d3, bb, sel1)[17, c(1, 2)] 
 a11 <- reg2(x1, y, d1, bb1, sel1)[1, c(1, 2)]
 a12 <- reg2(x2, y, d2, bb1, sel1)[1, c(1, 2)] 
 a13 <- reg2(x3, y, d3, bb1, sel1)[1, c(1, 2)] 
 m1 <- lm(y ~ x1+x2+x3, subset=sel1)
 p <- predict(m1, data.frame(x1=mean(x1), x2=mean(x2), x3=mean(x3)), se.fit=TRUE)
 a0 <- matrix(c(p$fit, sqrt(p$residual.scale^2/N + p$se.fit^2)), ncol=2)
 ans <- cbind(a161, a162, a163, a11, a12, a13, a0)
 ans}
 
do.sim <- function(reps=1000, ...){
 ans <- NULL
 for (j in 1:reps) ans <- rbind(ans, one.sim(...))
 data.frame(ans, row.names=NULL)}
 
#ans1 <- do.sim() # the simulation takes several hours
ans1 <- do.sim(reps=3)   ##tiny version of the simulation
#write.table(ans1, "sim1.out", row.names=FALSE, col.names=TRUE) # saves results
#summarize results
#ans1 <- read.table("sim1.out", header=TRUE)

sum.sim4 <- function(ans1, y=Spd1){
     true <- mean(y)
     groupsum <- function(x, true){c(mean(x[, 1]), true, 
                  sqrt(mean(x[, 2]^2)), sd(x[, 1]))}
     out <- NULL
     for (j in 0:6){out <- rbind(out, groupsum(ans1[, c(2*j+1, 2*j+2)], true))}
     out <-data.frame(out)
     row.names(out) <- c("16 bins S2",  "16 bins S3",  "16 bins S4", 
     "1 bin S2",  "1 bin S3",  "1 bin S4",  "4 references")
     names(out) <- c("Est mean",  "True mean",  "Est Ave SE",  "SE of means")
     out
}


sum.all<- function(ans1, y=Spd1, ltype=2){
     true <- mean(y)
     op<-par(mfrow=c(3, 2), mar=c(4, 3, 0, .5)+.1, mgp=c(2, 1, 0)) 
     hist(ans1[, 7],  xlab="Reference=Spd2,  ignore bins", xlim=c(7.2, 7.7), main="")
     abline(v=true, lty=ltype)
     hist(ans1[, 8],  xlab="Reference=Spd2,  ignore bins,  SE", xlim=c(.058, .084), main="")
     abline(v=sd(ans1[, 7]), lty=ltype)
     hist(ans1[, 1],  xlab="Reference=Spd2,  16 bins", xlim=c(7.2, 7.7), main="")
     abline(v=true, lty=ltype)
     hist(ans1[, 2],  xlab="Reference=Spd2,  16 bins,  SE", xlim=c(.058, .084), main="")
     abline(v=sd(ans1[, 1]), lty=ltype)
     hist(ans1[, 13],  xlab="Three references", xlim=c(7.2, 7.7), main="")
     abline(v=true, lty=ltype)
     hist(ans1[, 14],  xlab="Three references,  SE", xlim=c(.058, .084), main="")
     abline(v=sd(ans1[, 13]), lty=ltype)
     par(op)
}

# Fig. 10.1 "../EPS-figs/wmbig.eps"
sum.all(ans1)

sum.sim4(ans1)
