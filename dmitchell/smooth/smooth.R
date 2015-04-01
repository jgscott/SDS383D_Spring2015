# Daniel Lewis Mitchell
# Spring 2015

# kernel smoother
#   y is a vector of noisy observations
#   x is a vector of observation locations
#   kernel is a kernel function
#   bandwidth is a kernel bandwidth (optional)
smooth <- function(y, x, kernel=dnorm, bandwidth=NULL){
  n <- length(y)

  # smoothing matrix for bandwidth h
  hatmat <- function(h){
    H <- matrix(0, nrow=n, ncol=n)
    for(i in 1:n){
      d <- x-x[i]
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
    h <- -log(u)
    H <- hatmat(h)
    yhat <- H%*%y
    sum(((y-yhat)/(1-diag(H)))^2)
  }

  # bandwidth and predictions
  h <- if(is.null(bandwidth)) -log(optimize(objective,0:1)$min) else bandwidth
  H <- hatmat(h)
  yhat <- H%*%y

  # estimate residual variance
  r <- y-yhat
  sigsq <- sum(r*r)/(n-2*sum(diag(H))-sum(diag(t(H)%*%H)))

  # pointwise confidence
  sd <- sqrt(sigsq*rowSums(H*H))

  # return a smooth object
  structure(list(y=y,x=x,yhat=yhat,h=h,sd=sd), class="smooth")
}

plot.smooth <- function(object){
  with(object, {
    k <- order(x)
    hi <- yhat+2*sd
    lo <- yhat-2*sd
    plot(x,y,type="n")
    polygon(c(x[k],rev(x[k])), c(lo[k],rev(hi[k])), col="#DDDDFF", border=NA)
    lines(x[k], yhat[k], lwd=2, col="#0000FF")
    points(x,y)
  })
}

fitted.smooth <- function(object) object$yhat

residuals.smooth <- function(object) object$y - object$yhat

demo <- function(){
  utils <- read.csv("utilities.csv")
  s <- with(utils, smooth(y=gasbill/billingdays, x=temp))
  plot(s)
  plot(fitted(s), resid(s), xlab="fitted value", ylab="residual")
  abline(h=0, lty=3)
  invisible(s)
}
