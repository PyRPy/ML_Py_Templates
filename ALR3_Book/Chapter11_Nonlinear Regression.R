# Applied Linear Regression, Third edition
# Chapter 11 Nonlinear Regression
# October 14, 2004; revised January 2011 for alr3 Version 2.0,  R only
require(alr3)

# turkey data
if(is.null(version$language) == FALSE) data(turk0)

# Fig. 11.1 "../EPS-figs/nturkey1.eps"
plot(Gain ~ A, turk0, xlab="Amount (percent of diet)", ylab="Weight gain (g)")

n1 <- nls( Gain ~ th1 + th2*(1-exp(-th3*A)), 
           data=turk0, start=list(th1=620, th2=200, th3=10))

# Fig. 11.2 "../EPS-figs/nturkey2.eps"
with(data=turk0, 
plot(A, Gain, xlab="Amount (percent of diet)", ylab="Weight gain (g)"))
x <- (0:44)/100
lines(x, predict(n1, data.frame(A=x)))

# page 243
# Lack of fit test
p1 <- lm(Gain ~ as.factor(A), turk0)
anova(n1, p1)

# turkeys, with three sources
#turkey <- read.table("data/turkey.txt", header=TRUE)
tdata <- turkey
# create the indicators for the categories of S
tdata$S1 <- tdata$S2 <- tdata$S3 <- rep(0,dim(tdata)[1])
tdata$S1[tdata$S==1] <- 1
tdata$S2[tdata$S==2] <- 1
tdata$S3[tdata$S==3] <- 1
# fit the models using m as weights
# common regressions
m4 <- nls( Gain ~ th1 + th2*(1-exp(-th3*A)),
           data=tdata, start=list(th1=620, th2=200, th3=10), weights=m)
# most general
m1 <- nls( Gain ~ S1*(th11 + th21*(1-exp(-th31*A)))+
                   S2*(th12 + th22*(1-exp(-th32*A)))+
                   S3*(th13 + th23*(1-exp(-th33*A))),
           data=tdata,start= list(th11=620, th12=620, th13=620,
                                  th21=200, th22=200, th23=200,
                                  th31=10, th32=10, th33=10), weights=m)
# common intercept
m2 <- nls(Gain ~ th1 +
                  S1*(th21*(1-exp(-th31*A)))+
                  S2*(th22*(1-exp(-th32*A)))+
                  S3*(th23*(1-exp(-th33*A))),
          data=tdata,start= list(th1=620,
                              th21=200, th22=200, th23=200,
                              th31=10, th32=10, th33=10), wieghts=m)
# common intercept and asymptote
m3 <- nls( Gain ~ th1 + th2 *(
                   S1*(1-exp(-th31*A))+
                   S2*(1-exp(-th32*A))+
                   S3*(1-exp(-th33*A))),
          data=tdata, weights=m,
          start= list(th1=620, th2=200, th31=10, th32=10, th33=10))
anova(m4, m2, m1)
anova(m4, m3, m1)

sspe <- sum(tdata$SD^2*(tdata$m-1))
dfpe <- sum(tdata$m)
s2pe <- sspe/dfpe
sspe; dfpe; s2pe

# Fig. 11.3 
with(tdata, 
  plot(A, Gain, pch=S))
xx <- seq(0, .44, length=100)
# set m=1 in predict because of the weights
lines(xx, predict(m2, data.frame(m=1, A=xx, S1=rep(1, 100), 
         S2=rep(0, 100), S3=rep(0, 100))), lty=1)
lines(xx, predict(m2, data.frame(m=1, A=xx, S1=rep(0, 100), 
         S2=rep(1, 100), S3=rep(0, 100))), lty=2)
lines(xx, predict(m2, data.frame(m=1, A=xx, S1=rep(0, 100), 
         S2=rep(0, 100), S3=rep(1, 100))), lty=3)


# segmented regression example
smod <- C ~ th0 + th1*(pmax(0, Temp - gamma))
s1 <- nls(C ~ th0 + th1*(pmax(0, Temp - gamma)),
          data=segreg, start=list(th0=70, th1=.5, gamma=40))

# Fig. 11.4 
with(segreg, plot(Temp, C, xlab="Mean Temperature (F)", ylab="KWH/Day"))
lines(seq(8, 82, length=200),
     predict(s1, list(Temp=seq(8, 82, length=200))))
summary(s1)

# bootstrap uses bootCase written originally by Lexin Li, and simplified
# by S. Weisberg for use with the book.  It works for any regression object
# fix the random.seed, so I'll always get the same answer:
set.seed(10131985)
s1.boot <- bootCase(s1,B=99) 
# Fig. 11.5 "../EPS-figs/segreg2.eps"
scatterplotMatrix(s1.boot, diagonal="histogram",
       col=palette(),
       var.labels=c(expression(theta[1]), expression(theta[2]),
                expression(gamma)),
       ellipse=FALSE,smooth=TRUE,level=c(.90)) 
s1.boot.summary <- data.frame(rbind(
     apply(s1.boot, 2, mean),
     apply(s1.boot, 2, sd),
     apply(s1.boot, 2, function(x){quantile(x, c(.025, .975))})))
row.names(s1.boot.summary) <- c("Mean", "SD","2.5%", "97.5%")
colnames(s1.boot.summary) <- c("theta 1", "theta 2", "gamma") # remove '.'
s1.boot.summary
  
set.seed(3241944)
n1.boot <- bootCase(n1, B=99)  # tiny bootstrap
# Fig. 11.6 
scatterplotMatrix(n1.boot[ , n1.boot[2, ]<250],diagonal="histogram",
   col=palette(),lwd=0.7,pch=".",
   var.labels=c(expression(theta[1]),expression(theta[2]),
            expression(theta[3])),
   ellipse=FALSE,smooth=TRUE,level=c(.90))
  

# Artificial data, one predictor only
set.seed(101385)  # always use the same random numbers
b <- 1.25
n <- 25
sig <- .40
gdata <- function(n1=n, beta=b, sigma=sig) {
  x <- runif(n1)
  y <- round(exp(x*beta) + sigma*rnorm(n1),2)
  x <- round(x, 2)
  data.frame(x=x, y=y)}
d <- gdata()
plot(d$x, d$y)
lines((0:100)/100,exp(b*(0:100)/100))

RSS <- function(beta) { sum( (d$y - exp(beta*d$x))^2 ) }
RSSf <- function(beta) {
 ans <- NULL
 for (b in beta) ans <- c(ans, RSS(b))
 ans}
b1 <- (0:200)/100
plot(b1,-(n/2)*log(RSSf(b1)), type="l")

# The gauss newton algorithm
# starting value
b0 <- 2.7
fit <- exp(b0*d$x)
deriv <- d$x * fit
res <- d$y - fit + b0*deriv
lm(res ~ deriv-1)
