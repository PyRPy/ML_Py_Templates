# Applied Linear Regression,  Third edition
# Chapter 12 Logistic Regression
# October 14,  2004; revised January 2011 for alr3 Version 2.0,   R only

require(alr3)
################################################################
# Balsam fir blowdown 
# blowBF
# page 255
m1 <- glm(y ~ logb(D, 2), family=binomial(), data=blowBF)
summary(m1)

# Table 12.2
m2 <- glm(y ~ logb(D, 2) + S, family=binomial(), data=blowBF)
m3 <- update(m2,  ~ . + logb(D, 2):S)
summary(m2)
summary(m3)

# Fig. 12.1 "../EPS-figs/blow1.eps"
op<- if(is.null(version$language) == FALSE)
          par(mfrow=c(2, 1), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0), cex.lab=1.0, cex=0.7) else
          par(mfrow=c(2, 1))
with(blowBF, plot(jitter(logb(D, 2), amount=.05), 
        jitter(blowBF$y, amount=0.02), 
        xlab=expression(paste("(a) ", Log[2](Diameter))), 
        ylab="Blowdown indicator"))
#lines(smooth.spline(logb(D, 2), y), lty=3)
abline(lm(y ~ logb(D, 2), data=blowBF), lty=1)
xx <- with(blowBF, seq(min(D), max(D), length=100))
lo.fit <- loess(y ~ logb(D, 2), data=blowBF, degree=1)
lines(logb(xx, 2), predict(lo.fit, data.frame(D=xx)), lty=3)
lines(logb(xx, 2), predict(m1, data.frame(D=xx), type="response"), lty=2)

library(sm)
with(blowBF, sm.density.compare(logb(D, 2), y, lty=c(1, 2), 
            xlab=expression(paste("(b) ", log[2](D)))))  
legend("topright", inset=.02, legend=c("Y=0", "Y=1"), lty=c(1, 2))
par(op)


with(blowBF,{
     plot(logb(D, 2), S, col=c("red", "black")[y + 1], pch=y + 1)  
     points(logb(D[y==1], 2), S[y==1], col=2, pch=2)
    } )


logistic <- function(x) { 1/(1 + exp(-x)) }
xx <- seq(-6, 6, length=100)
# Fig. 12.2 "../EPS-figs/logistic.eps"
plot(xx, logistic(xx), type="l", 
           xlab=expression(paste(beta, "'", bold(x))), ylab="Probability") 


s1 <-summary(m1)
s1$coef
print(paste("Deviance:", round(s1$deviance, 2)))
print(paste("Pearson's X^2:", round(sum(residuals(m1, type="pearson")^2), 2)))


# Fig. 12.3 "../EPS-figs/blow1a.eps"
op <- par(mfrow=c(2, 1), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0), 
          cex.lab=1.0, cex=0.95) 
with(blowBF,
  sm.density.compare(S, y, lty=c(1, 2), xlab=expression(paste("(a) ", S)))) 
legend("topright", inset=.02, legend=c("Y=0", "Y=1"), lty=c(1, 2), pch=c(1, 2))  
with(blowBF,  {
  plot(jitter(logb(D, 2), amount=.04), S, pch=y + 1, cex=0.5, 
          xlab=expression(paste("(b) ",  log[2](D)))) 
  points(jitter(logb(D[y==0], 2), factor=.5), S[y==0], pch=2)
})
legend("topright", legend=c("Y=0", "Y=1"), pch=c(1, 2), cex=0.8)
par(op)

m2 <- update(m1,  ~ . + S)
m3 <- update(m2,  ~ . + S:logb(D, 2))

# Fig. 12.4 "../EPS-figs/blowc.eps"
op <- par(mfrow=c(2, 1), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0), 
      cex.lab=1.0, cex=0.95) 
with(blowBF, {
xa <- seq(min(D), max(D), len=99)
ya <- seq(.01,  .99, len=99)
za <- matrix(nrow=99, ncol=99)
for (i in 1:99) {
 za[, i] <- predict(m2, data.frame(D=rep(xa[i], 99), S=ya), type="response")}
if(is.null(version$language) == FALSE){
  contour(logb(xa, 2), ya, za, xlab=expression(paste("(b) ", log[2](D))), ylab="S")
  points(jitter(logb(D, 2), amount=.04), S, pch=y + 1, cex=.5)} else {
  contour(logb(xa, 2), ya, za, xlab="(b) log[2](D)", ylab="S")
  points(jitter(logb(D[y==0], 2), factor=.4), S[y==0], pch=1, cex=.5)
  points(jitter(logb(D[y==1], 2), factor=.4), S[y==1], pch=2, cex=.5)}
# second model with interaction
za <- matrix(nrow=99, ncol=99)
for (i in 1:99) {
 za[, i] <- predict(m3, data.frame(D=rep(xa[i], 99), S=ya), type="response")}
if(is.null(version$language) == FALSE){
  contour(logb(xa, 2), ya, za, xlab=expression(paste("(b) ", log[2](D))), ylab="S")
  points(jitter(logb(D, 2), amount=.04), S, pch=y + 1, cex=.5)
  } else {
  contour(logb(xa, 2), ya, za, xlab="(b)log[2](D)", ylab="S")
  points(jitter(logb(D[y==0], 2), factor=.4), S[y==0], pch=1, cex=.5)
  points(jitter(logb(D[y==1], 2), factor=.4), S[y==1], pch=2, cex=.5)}
})
par(op)


# Fig. 12.5 
xx <- seq(0, 1, length=100)
op <- par(mfrow=c(1, 2), mar=c(4, 3, 0, .5) + .1, mgp=c(2, 1, 0), cex=0.6) 
plot(xx, exp(.4009 + 4.9098*xx), type="l",  xlab="(a) S", ylab="Odds multiplier") 
with(blowBF, xx <- seq(min(logb(D, 2)), max(logb(D, 2)), length=100))
plot(2^(xx), exp(coef(m3)[3]/10 + coef(m3)[4]*xx/10), type="l",  xlab="(b) D", ylab="Odds multiplier") 
par(op)

summary(m2)
summary(m3)
print(paste("Pearson's X^2:", round(sum(residuals(m3, type="pearson")^2), 2)))
anova(m1, m2, m3, test="Chisq")

anova(m1, m3, test="Chisq")
m0 <- update(m1,  ~ . -logb(D, 2))
anova(m0, m1, m2, m3, test="Chisq") 


#############################################################################
# Titanic data
dt <- titanic
head(titanic)
mysummary <- function(m){c(df=m$df.residual, G2=m$deviance, 
                           X2=sum(residuals(m, type="pearson")^2) )}

m1 <- glm(cbind(Surv, N-Surv) ~ Class + Age + Sex,  data=titanic,  family=binomial())
m2 <- update(m1,  ~ . + Class:Sex)
m3 <- update(m2,  ~ . + Class:Age)
m4 <- update(m3,  ~ . + Age:Sex)
m5 <- update(m4,  ~ Class:Age:Sex)

ans <- mysummary(m1)
ans <- rbind(ans, mysummary(m2))
ans <- rbind(ans, mysummary(m3))
ans <- rbind(ans, mysummary(m4))
ans <- rbind(ans, mysummary(m5))
row.names(ans) <- c( m1$formula,  m2$formula,  m3$formula, m4$formula, 
   m5$formula) 
ans
