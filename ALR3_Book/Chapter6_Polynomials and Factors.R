# Applied Linear Regression, Third edition
# weis06.R, chapter 6 Polynomials and Factors
# this chapter needs to be further studied. quite a lot of points are not clear.
# October 14, 2004; revised January 2011 for alr3 Version 2.0, R only
require(alr3)

# quadratic curves in one variable
x <- seq(-1, 1, length=51)
y1 <- -x^2 + 1
y2 <- x^2
z <- data.frame(x=c(x, x), y=c(y1, y2), g=rep(c("(a)", "(b)"), c(51, 51)))
require(lattice)  

# Fig. 6.1 "../EPS-figs/quad.eps"
xyplot(y ~ x|g, z, xlab="X", ylab="E(Y|X)", type="l", aspect=1, 
           between=list(x=.25), 
           panel = function(x, y){
             panel.xyplot(x, y, type="l")
             panel.abline(v=.25,  lty=2)
             panel.abline(v=.75,  lty=2)}, 
           scales=list(draw=FALSE))

# Oehlert's cake baking data
# Fig. 6.2 
head(cakes)
with(cakes,
   plot(jitter(X1), jitter(X2), xlab=expression(X[1]), ylab=expression(X[2]))) 

m1 <- lm(Y ~ X1 + X2 + I(X1^2) + I(X2^2) + X1:X2, data=cakes)
m0 <- update(m1, ~ block + .)

# Fig. 6.3 "../EPS-figs/cake2.eps"
oldpar<-par(mfrow=c(1, 2), mar=c(4, 3, 0, .5) + .1, mgp=c(2, 1, 0))
# part (a)
with(cakes,
   plot(X1, Y, type="n", xlab=expression(paste("(a)  ", X[1]))) )
X1new <- seq(32, 38, len=50)
lines(X1new, predict(m1, newdata=data.frame(X1=X1new, X2=rep(340, 50))))
lines(X1new, predict(m1, newdata=data.frame(X1=X1new, X2=rep(350, 50))))
lines(X1new, predict(m1, newdata=data.frame(X1=X1new, X2=rep(360, 50))))
text(34, 4.7, "X2=340", adj=0, cex=0.7)
text(32.0, 5.7, "X2=350", adj=0, cex=0.7)
text(32.0, 7.6, "X2=360", adj=0, cex=0.7)
# part (b)
with(cakes,
   plot(X2, Y, type="n", xlab=expression(paste("(b)  ", X[2]))))
X2new <- seq(330, 370, len=50)
lines(X2new, predict(m1, newdata=data.frame(X1=rep(33, 50), X2=X2new)))
lines(X2new, predict(m1, newdata=data.frame(X1=rep(35, 50), X2=X2new)))
lines(X2new, predict(m1, newdata=data.frame(X1=rep(37, 50), X2=X2new)))
text(342, 4,   "X1=33", adj=0, cex=0.7)
text(335, 4.55, "X1=35", adj=0, cex=0.7)
text(336, 7.3, "X1=37", adj=0, cex=0.7)
par(oldpar)

m3 <- update(m1,   ~ .-X1:X2)
# Fig. 6.4 "../EPS-figs/cake3.eps"
oldpar<-par(mfrow=c(1, 2), mar=c(4, 3, 0, .5) + .1, mgp=c(2, 1, 0))
# (a)
with(cakes,
   plot(X1, Y, type="n", xlab=expression(paste("(a)  ", X[1]))))
X1new <- seq(32, 38, len=50)
lines(X1new, predict(m3, newdata=data.frame(X1=X1new, X2=rep(340, 50))))
lines(X1new, predict(m3, newdata=data.frame(X1=X1new, X2=rep(350, 50))))
lines(X1new, predict(m3, newdata=data.frame(X1=X1new, X2=rep(360, 50))))
text(33, 4.3, "X2=340", adj=0, cex=0.7)
text(32.0, 7.2, "X2=350", adj=0, cex=0.7)
text(34.0, 7.1, "X2=360", adj=0, cex=0.7)
# (b)
with(cakes,
   plot(X2, Y, type="n", xlab=expression(paste("(b)  ", X[2]))))
X2new <- seq(330, 370, len=50)
lines(X2new, predict(m3, newdata=data.frame(X1=rep(33, 50), X2=X2new)))
lines(X2new, predict(m3, newdata=data.frame(X1=rep(35, 50), X2=X2new)))
lines(X2new, predict(m3, newdata=data.frame(X1=rep(37, 50), X2=X2new)))
text(340, 4.3,   "X1=33", adj=0, cex=0.7)
text(336, 7.0, "X1=35", adj=0, cex=0.7)
text(346, 7.3, "X1=37", adj=0, cex=0.7)
par(oldpar)

# illustrating poly in the primer
x <- 1:5
print(xmat <- cbind(rep(1, 5), x, x^2, x^3))
qr.Q(qr(xmat))
poly(x, 3)

# delta method stuff in the supplement
m4 <- lm(Y  ~  X2  +  I(X2^2),  data=cakes)
b0 <- coef(m4)[1]
b1 <- coef(m4)[2]
b2 <- coef(m4)[3]
Var <- vcov(m4)
xm <- "-b1/(2*b2)"
xm.expr <- parse(text=xm)
xm.expr
eval(xm.expr)
derivs <- c(D(xm.expr, "b0"), D(xm.expr, "b1"), D(xm.expr, "b2"))
derivs
eval.derivs<-c(eval(D(xm.expr, "b0")), eval(D(xm.expr, "b1")), 
               eval(D(xm.expr, "b2")))
eval.derivs
sqrt(t(eval.derivs) %*% Var %*% eval.derivs)

deltaMethod(m4, "-b1/(2*b2)")

#sleep1
head(sleep1)
sleep1$D <- factor(sleep1$D) # D is 'danger' level
a1 <- lm(TS ~ D, sleep1, na.action=na.omit)
a0 <- update(a1,  ~ .-1)
compareCoefs(a1, a0)                    

# Fig. 6.5 "../EPS-figs/sleep11.eps"
plot(sleep1$D, sleep1$TS, xlab="Danger index" , ylab="Total sleep1 (h)") # in R it is 'boxplot' for...
anova(a0)
anova(a1)


m1 <- lm(TS ~ logb(BodyWt, 2)*D, sleep1, na.action=na.omit)
m2 <- update(m1,   ~ D  +  logb(BodyWt, 2))
m4 <- update(m1,   ~ logb(BodyWt, 2))
m3 <- update(m1,   ~ logb(BodyWt, 2):D)
compareCoefs(m1, m2, m4, m3, se=FALSE)

# all have the same Residual sum of squares:
a1<-anova(m4, m1)
a2<-anova(m3, m1)
a3<-anova(m2, m1)
a1[2, ]; a2[2, ]; a3[2, ]

# Figure with lattice fitting all four models.
n1 <- lm(TS ~ -1 + D + D:logb(BodyWt, 2), sleep1, na.action=na.omit)
n2 <- update(m1,   ~ -1 + D  +  logb(BodyWt, 2))
n4 <- update(m1,   ~ logb(BodyWt, 2))
n3 <- update(m1,   ~ logb(BodyWt, 2):D)

d <- rbind(sleep1, sleep1, sleep1, sleep1)
labels <- c("(a) General",  "(b) Parallel", "(c) Common intercept",  "(d) Common regression")
d$f <- rep(c(1, 2, 3, 4),  rep(62, 4))
d$fig <- rep(labels,  rep(62, 4))
# Fig. 6.6 "../EPS-figs/sleeplines.eps"
xyplot(TS ~ logb(BodyWt, 2)|fig, group=D,  d,  subscripts=TRUE, ID=d$f, 
      as.table=TRUE,  key=simpleKey(as.character(1:5), FALSE, FALSE, T), 
      xlab=expression(log[2](BodyWt)),  cex=.75,  between=list(x=.32, y=.32), 
      panel = function(x, y, groups, subscripts, ID){
          panel.superpose(x, y, groups=groups, subscripts=subscripts, 
              pch=as.character(1:5))
      g <- unique(ID[subscripts]) # which panel
      if(g==1) {panel.abline(coef(n1)[c(1, 6)], lty=1)
                panel.abline(coef(n1)[c(2, 7)], lty=2)
                panel.abline(coef(n1)[c(3, 8)], lty=3)
                panel.abline(coef(n1)[c(4, 9)], lty=4)
                panel.abline(coef(n1)[c(5, 10)], lty=5)}
      if(g==3) {panel.abline(coef(n3)[c(1, 2)], lty=1)
                panel.abline(coef(n3)[c(1, 3)], lty=2)
                panel.abline(coef(n3)[c(1, 4)], lty=3)
                panel.abline(coef(n3)[c(1, 5)], lty=4)
                panel.abline(coef(n3)[c(1, 6)], lty=5)}
      if(g==2) {panel.abline(coef(n2)[c(1, 6)], lty=1)
                panel.abline(coef(n2)[c(2, 6)], lty=2)
                panel.abline(coef(n2)[c(3, 6)], lty=3)
                panel.abline(coef(n2)[c(4, 6)], lty=4)
                panel.abline(coef(n2)[c(5, 6)], lty=5)}
      if(g==4) {panel.abline(coef(n4)[c(1, 2)], lty=1)}
      })



# Partial 1D models
# Fig. 6.7 
# relabel sex  from 0 and1 to male and female
head(ais)
ais$Sex <- factor(c("male", "female")[ais$Sex + 1])
scatterplotMatrix(~ LBM + Ht +Wt + RCC|Sex, data=ais)
pairs(ais[, c(4, 2, 3, 5)]) 


m <- pod(LBM ~ Ht + Wt + RCC,  data=ais,  group=Sex)
anova(m)

# Not in the text "../EPS-figs/supp61.eps"
plot(m, pch=c("m", "f"), colors=c("red", "black"))

# Random coef. models
library(nlme)
# Fig. 6.9 "../EPS-figs/chloride1.eps"
xyplot(Cl ~ Month|Type,  group=Marsh,  data=chloride, ylab="Cl (mg/l)", 
     xlab="Month number", type=c("g", "p", "l"))
m1 <- lme(Cl ~ Month + Type,  data=chloride,  random= ~ 1 + Type|Marsh)
m2 <- update(m1,  random= ~ 1|Marsh)
anova(m1, m2)
intervals(m2)
