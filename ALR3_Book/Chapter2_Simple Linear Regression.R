# Applied Linear Regression, Third edition
# Chapter 2
# October 14, 2004; revised January 2011 for alr3 Version 2.0, R only

# Fig. 2.1 in the new edition
# R only
library(alr3)
x <- c(0, 4)
y <- c(0, 4)
plot(x, y, type="n", xlab="Predictor = X", ylab="E(Y|X=x)")
abline(.8, 0.7)
x<-c(2, 3, 3)
y<-c(2.2, 2.2, 2.9)
lines(x, y)
lines(c(0, 0), c(0, .8), lty=2)
lines(c(0, 4), c(0, 0), lty=2)
text(3.05, 2.5, expression(beta[1] == Slope), adj=0)
text(.05, .4, expression(beta[0] == Intercept), adj=0)
text(2.5, 1.8, "1")


# Fig. 2.2.  
plot(c(0, .4),c(150, 400), type="n", xlab="X", ylab="Y")
abline(135, 619.712, lty=2)
x <- seq(0,.4, length=200)
y <- 178.581 + 79.3528*x + 1369.44*x*x
lines(x, y)
lines( c(.325, .325), c(336.406, 349.017))
text(.07, 165, "Simple regression mean function", adj=0, cex=0.6)
text(.34, 380, "True mean function", adj=1, cex=0.6)
text(.305, 300, "Fixed lack of fit error", adj=0, cex=0.6)
arrows(.305, 302, .323, 333, .06)


# Fig. 2.3
x<-c(0.45, 3.97, 1.12, 4.21, -0.23, 1.56, -0.01, 2.67, 1.51, 
     0.93, 2.08, 3.50,4.14, 1.92, 2.11, 3.25, 0.67, 0.68, -0.22, 3.96)
y<-c(-1.24, 4.87, 1.32, 4.14, -0.62, 0.66, 0.61, 3.27, 1.97, 0.40, 2.49, 
     4.00, 3.11, 2.08, 1.97, 2.70, 1.50, 1.77, -0.26, 2.11)
m1 <- lm(y ~ x)
yhat <- predict(m1)
plot(x, y, type="p", xlab=expression(Predictor==X),
     ylab=expression(Response==Y))
abline(m1)
abline(.8, .7, lty=2)
text(1, -.4, "Residuals are the signed lengths of", cex=0.7, adj=0)
text(1, -.65, "the vertical lines", cex=0.7, adj=0)
arrows(.95, -.5, .46, -.3, length=.08)
arrows(1.1, -.15, 1.54, 1, length=.08)
for (i in 1:20){lines(list(x=c(x[i], x[i]),y=c(y[i], yhat[i])))}


# computations in text
# page - 24
#forbes
forbes1 <- forbes[,c(1,3)]
forbes1
fmeans <- colMeans(forbes1) 
fmeans # xbar and ybar

xbar <- fmeans[1] ; ybar <- fmeans[2]

# get SXX, SXY and SYY.  
fcov <- (17-1) * var(forbes1)
fcov

SXX <- fcov[1,1]; SXY<- fcov[1,2]; SYY <- fcov[2,2]
(betahat1 <- SXY/SXX)
(betahat0 <- ybar - betahat1*xbar)

(RSS <- SYY - SXY^2/SXX)
(SSreg <- SYY - RSS)
(R2 <- SSreg/SYY)

(sigmahat2 <- RSS/15) # Eq (2.7) , page 25
(sigmahat <- sqrt(sigmahat2))

a <- data.frame(list(values=c(xbar, ybar, SXX, SXY, SYY, SSreg, R2,
                     betahat1, betahat0, RSS, sigmahat2 ,sigmahat)),
    row.names=c("xbar", "ybar", "SXX", "SXY", "SYY", "SSreg", "R2", 
                     "betahat1", "betahat0", "RSS", 
                     "sigmahat2", "sigmahat"))
round(a, 5)

# Figure 2.4
# page 28
m0 <- update(m1, ~ . -x)  # intercept only
plot(x, y, type="p", xlab=expression(Predictor==X),
     ylab=expression(Response==Y))
abline(reg=m1)
abline(m0$coef[1],0)
text(2.9, 1.5, "Fit of (2.13)", adj=0)
text(0.0, -0.2, "Fit of (2.16)", adj=0)

# Forbes anova table
m1 <- lm(Lpres ~ Temp, data=forbes)
summary(m1)
anova(m1)


(betahat<-coef(m1))
(var <- vcov(m1))
(tval <- qt(1-.05/2, m1$df))
data.frame(Est = betahat, lower=betahat-tval*sqrt(diag(var)),
                          upper=betahat+tval*sqrt(diag(var))) 

confint(m1)
predict(m1)
predict(m1, newdata=data.frame(Temp=c(210,220)))
predict(m1, newdata=data.frame(Temp=c(210,220)),
 se.fit=TRUE, interval="prediction", level=.95)

residuals(m1)

# p-value for an test of beta_0 = 35:
s1 <- summary(m1)
(s1$coefficients[1,1]+35)/s1$coefficients[1,2]
2*pt((s1$coefficients[1,1]+35)/s1$coefficients[1,2],15)

(p200 <- predict(m1, data.frame(Temp=200),
        interval="prediction", level=.99))

#prediction of Pressure
10^(p200/100)

# predictions and fitted values for the heights data
m1 <- lm(Dheight ~ Mheight, data=heights)
new <- data.frame(Mheight=seq(55.4, 70.8, length=50))
pred.w.plim <- predict(m1, new, interval="prediction")
pred.w.clim <- predict(m1, new, interval="confidence")
# R does not use the Scheffe correction, we I need to fix up this
# last interval.
cf <- sqrt(2*qf(.975, 2, 1373))/qt(.975, 1373)
pred.w.clim[,2] <- -(pred.w.clim[, 1] - pred.w.clim[, 2])*cf +
                   pred.w.clim[, 1]
pred.w.clim[,3] <- (pred.w.clim[,3] - pred.w.clim[, 1])*cf +
                   pred.w.clim[,1]                   
# Fig. 2.5 
matplot(new$Mheight,cbind(pred.w.clim, pred.w.plim[, -1]),
             col=rep(1, 5),
             lty=c(2, 3, 3, 1, 1), type="l", ylab="Dheight",
             xlab="Mheight")


# residual plot for heights data.
m1 <- lm(Dheight ~ Mheight, data=heights)
# Fig. 2.6 
plot(predict(m1),residuals(m1),cex=.3,
      xlab="Fitted values",ylab="Residuals")
abline(h=0, lty=2)


# residual plot for forbes data 
m1 <- lm(Lpres ~ Temp, data=forbes)
# Fig. 2.7 
plot(predict(m1), residuals(m1),
     xlab="Fitted values", ylab="Residuals")
text(predict(m1)[12],residuals(m1)[12],labels="12",adj=-1)
abline(0,0)

# can be obtained using the car function residualPlots
residualPlots(m1, terms= ~ 1, fitted=TRUE, id.n=1, id.method="y")
# terms = ~ 1 suppresses all plots versus predictors
# fitted=TRUE includes residuals vs fitted values
# id.n=1 identifies one most extreme residual


# Table 2.5
m2 <- update(m1, subset=-12)
s1 <- summary(m1)
s2 <- summary(m2)
ans <-matrix(c(s1$coefficients[,1],s1$coefficients[,2],s1$sigma,
             s1$r.squared,
             s2$coefficients[,1],s2$coefficients[,2],s2$sigma,
             s2$r.squared),ncol=2,
             dimnames=list(c("betahat_0",
                             "betahat_1",
                             "se(betahat_0)",
                             "se(betahat_1)",
                             "sigmahat",
                             "R^2"),
                c("All data", "Delete case 12")))

round(ans, 5)

# Ft. Collins data
# page - 8 for some explation on the data, shows almost 'no' correlation, but here... p-value...
snow1 <- lm(Late ~ Early, data=ftcollinssnow)
summary(m1)
anova(m1)
