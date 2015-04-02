# Daniel Lewis Mitchell
# Spring 2015

# kernel smoother
#   y is a vector of noisy observations
#   x is a vector of observation locations
#   kernel is a kernel function
#   bw is the kernel bandwidth (optional)
# if the bandwidth is not supplied it will be determined by leave-one-out
# cross validation
smooth <- function(y, x, kernel=dnorm, bw=NULL){

  # smoothing matrix for prediction locations z and bandwidth h
  hatmat <- function(z, h){
    H <- matrix(0, nrow=length(z), ncol=length(y))
    for(i in 1:length(z)){
      d <- x-z[i]
      w <- kernel(d/h)/h
      s1 <- sum(w*d)
      s2 <- sum(w*d*d)
      w <- w*(s2-d*s1)
      H[i,] <- w/sum(w)
    }
    return(H)
  }

  # leave-one-out prediction error for bandwidth h = -log(u), 0 < u < 1
  objective <- function(u){
    H <- hatmat(x, -log(u))
    yhat <- H%*%y
    sum(((y-yhat)/(1-diag(H)))^2)
  }

  # select bandwidth
  h <- if(is.null(bw)) -log(optimize(objective,0:1)$min) else bw

  # estimate residual variance
  H <- hatmat(x, h)
  yhat <- H%*%y
  r <- y-yhat
  sigsq <- sum(r*r)/(length(y)-2*sum(diag(H))-sum(diag(t(H)%*%H)))
  se.fit <- sqrt(sigsq*rowSums(H*H))

  # return a smooth object
  result <- list(y=y,
                 x=x,
                 bandwidth=h,
                 sigsq=sigsq,
                 residuals=r,
                 fitted=yhat,
                 se.fit=se.fit,
                 predict=function(newx){
                   H <- hatmat(newx,h)
                   list(fit=as.vector(H%*%y), se.fit=sqrt(sigsq*rowSums(H*H)))
                 })
  structure(result, class="smooth")
}

fitted.smooth <- function(object) object$fitted

residuals.smooth <- function(object) object$residuals

predict.smooth <- function(object, newx, se.fit=FALSE){
  result <- object$predict(newx)
  if(se.fit) result else result$fit
}

plot.smooth <- function(object){
  with(object, {
    k <- order(x)
    upper <- fitted + 2*se.fit
    lower <- fitted - 2*se.fit
    conf.x <- c(x[k],x[rev(k)])
    conf.y <- c(lower[k], upper[rev(k)])
    plot(conf.x, conf.y, type="n", xlab="x", ylab="y")
    polygon(conf.x, conf.y, col="#DDDDFF", border=NA)
    points(x,y)
    points(x[k], fitted[k], type="l", lwd=2, col="#0000FF")
  })
}

demo <- function(){
  utils <- read.csv("utilities.csv")
  x <- with(utils, temp)
  y <- with(utils, gasbill/billingdays)
  s <- smooth(y, x)
  readline("Hit <Return> to see next plot: ")
  plot(s)
  readline("Hit <Return> to see next plot: ")
  plot(fitted(s), residuals(s), xlab="Fitted values", ylab="Residuals")
  abline(h=0, lty=3)
  readline("Hit <Return> to see next plot: ")
  newx <- runif(10, min=min(x), max=max(x))
  pr <- predict(s, newx, se.fit=TRUE)
  fit <- pr$fit
  se <- pr$se.fit
  plot(newx, fit, xlab="x", ylab="y")
  arrows(newx, fit-2*se, newx, fit+2*se, length=0.05, angle=90, code=3)
  invisible(s)
}
