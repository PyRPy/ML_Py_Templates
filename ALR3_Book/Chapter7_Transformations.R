# Applied Linear Regression,  Third edition
# Chapter 7 Transformations
# October 14,  2004; revised January 2011 for alr3 Version 2.0,  R only
require(alr3)

# computations using brain weight data
# Fig 7.1 "../EPS-figs/brains1.eps"
scatterplot(BrainWt ~ BodyWt, data=brains, id.n=3, smooth=FALSE, box=FALSE)

# Fig. 7.2 "../EPS-figs/brains2.eps"
oldpar<-if(is.null(version$language) == FALSE)
  par(mfrow=c(2, 2), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0), cex.lab=1.0, cex=0.6) else
  par(mfrow=c(2, 2), mai=c(.6, .6, .1, .1), mgp=c(2, 1, 0)) 

# inverse
plot(I(1/BrainWt) ~ I(1/BodyWt), data=brains, 
     xlab="1/BodyWt", ylab="1/BrainWt")
abline(lm(I(1/BrainWt) ~ I(1/BodyWt), brains))
with(brains, lines(lowess(I(1/BodyWt), I(1/BrainWt)), lty=2, col="blue"))

# log
plot(log(BrainWt) ~ log(BodyWt), data=brains, 
     xlab="log(BodyWt)", ylab="log(BrainWt)")
abline(lm(log(BrainWt) ~ log(BodyWt), brains))
with(brains, lines(lowess(log(BodyWt), log(BrainWt)), lty=2, col="blue"))

# cube root
plot(I(BrainWt^(1/3)) ~ I(BodyWt^(1/3)), data=brains, 
     xlab="BodyWt^(1/3)", ylab="BrainWt^(1/3)")
abline(lm(I(BrainWt^(1/3)) ~ I(BodyWt^(1/3)), brains))
with(brains, lines(lowess(I(BodyWt^(1/3)), I(BrainWt^(1/3))), lty=2, col="blue"))

# squareroot
plot(I(BrainWt^(1/2)) ~ I(BodyWt^(1/2)), data=brains, 
     xlab="BodyWt^(1/2)", ylab="BrainWt^(1/2)")
abline(lm(I(BrainWt^(1/2)) ~ I(BodyWt^(1/2)), brains))
with(brains, lines(lowess(I(BodyWt^(1/2)), I(BrainWt^(1/2))), lty=2, col="blue"))
par(oldpar)

#ufcwc
lam <-c(1, 0, -1)
new <- with(ufcwc,seq(min(Dbh), max(Dbh), length=100))
# always inspect the data first, even before plotting !
head(ufcwc)

# Fig 7.3 
with(ufcwc, plot(Dbh, Height))
with(ufcwc, 
  for (j in 1:3){
    m1 <- lm(Height ~ bcPower(Dbh, lam[j]) )
    lines(new, predict(m1, data.frame(Dbh=new)), lty=j, col=j, lwd=2)})
legend("bottomright", inset=.02, legend=as.character(lam), 
                     lty = 1:3, col=1:3, xjust = 1, yjust = 1)

# Not in text but in primer.  This is the same plot, but it uses
# invTranPlot in the car package
with(ufcwc, 
  invTranPlot(Dbh, Height, lambda = c(-1,  0,  1)))


# Fig. 7.4 
scatterplot(Height ~ log2(Dbh), data=ufcwc, smooth=FALSE, box=FALSE)
scatterplot(Height ~ log2(Dbh), data=ufcwc, smooth=F, box=F) # what's difference when smooth/box changes

# Scatterplot matrix stuff
# highway data
# transform X 
cols <- c(12, 8, 1, 2, 7, 10, 5)
# Fig. 7.5 
scatterplotMatrix( ~ Rate + Len + ADT + Trks + Slim + Shld + Sigs, 
   data=highway, smooth=FALSE)

# New syntax for powerTransform, see the help page.
summary(ans <- powerTransform(cbind(Len, ADT, Trks, Shld, Sigs) ~ 1,
   data=highway, family="yjPower"))

# Repalce Sigs by Sigs1 which is always positive and use Box-Cox family
highway$Sigs1 <- (round(highway$Sigs*highway$Len) + 1)/highway$Len
ans <- powerTransform(cbind(Len, ADT, Trks, Shld, Sigs1) ~ 1, 
                      data=highway)
summary(ans)

plot(ans, family="power") # not working
# Error in xy.coords(x, y, xlabel, ylabel, log) : 
#   'x' and 'y' lengths differ

testTransform(ans, lambda=c(0, 0, 0, 1, 0)) # same as in 'summary' output
coef(ans) # returns estiamted transformation parameters
coef(ans, round=TRUE) # rounds to 'nice' values

# Fig. 7.7
plot(ans) # not working as one above

# add transformed values to data frame
highway <- cbind(highway, basicPower(ans$y, coef(ans, round=TRUE)))


# assume X transformed,  and transform Y
# Fig. 7.8 "../EPS-figs/highway3.eps"
m2 <- lm(Rate ~ log(Len) + log(ADT) + log(Trks) + Slim + Shld + 
     log(Sigs1), data=highway)  # base e logs

with(highway, invTranPlot(Rate, predict(m2)))

# Box-Cox method
boxCox(m2,  xlab=expression(lambda[y]))
