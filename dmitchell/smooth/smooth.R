# Daniel Lewis Mitchell
# Spring 2015

# kernel smoother
#   y is a vector of noisy observations
#   x is a vector of observation locations
#   kernel is a kernel function
#   bandwidth is a kernel bandwidth (optional)
smooth <- function(y, x, kernel=dnorm, bandwidth=NULL){
  n <- length(y)

  # compute smoothing matrix for bandwidth h
  hmat <- function(h){
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

  # leave-one-out prediction error for bandwidth h
  objective <- function(h){
    H <- hmat(h)
    yhat <- H%*%y
    sum(((y-yhat)/(1-diag(H)))^2)
  }

  # bandwidth and predictions
  h <- bandwidth
  if(is.null(h)) h <- optimize(objective,c(0,max(x)-min(x)))$min
  H <- hmat(h)
  yhat <- H%*%y

  # pointwise confidence
  r <- y-yhat
  sigsq <- sum(r*r)/(length(y)-2*sum(diag(H))-sum(diag(t(H)%*%H)))
  sd <- sqrt(sigsq*rowSums(H*H))

  # return a smooth object
  structure(list(y=y,x=x,yhat=yhat,h=h,sd=sd), class="smooth")
}

plot.smooth <- function(X){
  with(X, {
    hi <- yhat+2*sd
    lo <- yhat-2*sd
    plot(x,y,type="n")
    polygon(c(x,rev(x)), c(lo,rev(hi)), col="#DDDDFF", border=NA)
    lines(x, yhat, lwd=2, col=4)
    points(x,y)
  })
}

demo <- function(){
  utils <- read.csv("utilities.csv")
  utils <- utils[order(utils$temp),]
  y <- with(utils, gasbill/billingdays)
  x <- with(utils, temp)
  s <- smooth(y, x)
  plot(s)
  invisible(s)
}
demo()
