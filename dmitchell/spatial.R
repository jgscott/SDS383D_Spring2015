# Daniel Lewis Mitchell
# Spring 2015

# squared exponential covariance
#   d is a matrix of distances
sqexp <- function(d, tausq, b){ d <- d/b; tausq*exp(-d*d/2) }

# matern 5/2 covariance
#   d is a matrix of distances
matern <- function(d, tausq, b){ d <- sqrt(5)*d/b; tausq*(1+d+d*d/3)*exp(-d) }

# spatial smoothing
#   y is a vector of observations
#   x is a SpatialPoints object with observation locations (longlat)
#   newx is a SpatialPoints object with prediction locations (longlat)
#   cov.fun is a covariance function
#   theta is null or a list of hyperparameters (tausq, b, sigsq)
#
# returns hyperparameters and the posterior mean and SD at prediction locations
smooth <- function(y, x, newx, cov.fun, theta=NULL, scale=TRUE){
  n <- length(x)
  m <- length(newx)

  # calculate distance to each point in x
  s <- matrix(0, nrow=n, ncol=n+m)
  for(i in 1:n) s[,i] <- spDistsN1(x, x[i,], longlat=T)
  for(i in 1:m) s[,n+i] <- spDistsN1(x, newx[i,], longlat=T)

  # negative log marginal likelihood for empirical bayes
  objective <- function(u, z){
    tausq <- exp(u[1]); b <- exp(u[2]); sigsq <- exp(u[3])
    cmat <- cov.fun(s[1:n,1:n], tausq, b) + diag(sigsq, n)
    log(det(cmat)) + t(z) %*% solve(cmat) %*% z
  }

  # use empirical bayes to estimate hyperparameters if none given
  if(is.null(theta)){
    if(scale) z <- scale(y) else z <- y
    u <- exp(optim(c(0,0,0), function(u) objective(u,z))$par)
    theta <- list(tausq=u[1], b=u[2], sigsq=u[3])
  }

  # evaluate covariance function for all distances
  s <- cov.fun(s, theta$tausq, theta$b)

  # precision matrix
  s[1:n,1:n] <- solve(s[1:n,1:n]+diag(theta$sigsq,n))

  # posterior mean and standard deviation
  p <- matrix(0, nrow=m, ncol=2)
  for(i in 1:m){
    p[i,1] <- s[,n+i] %*% s[1:n,1:n] %*% y
    p[i,2] <- sqrt(theta$tausq - t(s[,n+i]) %*% s[1:n,1:n] %*% s[,n+i])
  }
  return(list(tausq=theta$tausq, b=theta$b, sigsq=theta$sigsq, mean=p[,1], sd=p[,2]))
}

demo <- function(){
  require(sp)
  require(raster)

  weather <- read.csv("weather.csv")
  coordinates(weather) <- ~lon+lat

  # create output points
  R <- brick(extent(weather), 200, 200, nl=4)
  newx <- rasterToPoints(R, spatial=TRUE)

  # smooth temperature
  sm <- smooth(weather$temperature, weather, newx, matern)
  R <- setValues(R, sm$mean, layer=1)
  R <- setValues(R, sm$sd, layer=2)

  # smooth pressure
  sm <- smooth(weather$pressure, weather, newx, matern)
  R <- setValues(R, sm$mean, layer=3)
  R <- setValues(R, sm$sd, layer=4)

  return(R)
}

library(rasterVis)
library(RColorBrewer)

spectral.colors <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
heat.colors <- colorRampPalette(rev(brewer.pal(9, "YlOrRd")))

R <- demo()
levelplot(R, layer=1, col.regions=spectral.colors, contour=T, margin=F, xlab="Longitude", ylab="Latitude", main="Temperature")
levelplot(R, layer=2, col.regions=heat.colors, contour=T, margin=F, xlab="Longitude", ylab="Latitude", main="Temperature standard deviation")
levelplot(R, layer=3, col.regions=spectral.colors, contour=T, margin=F, xlab="Longitude", ylab="Latitude", main="Pressure")
levelplot(R, layer=4, col.regions=heat.colors, contour=T, margin=F, xlab="Longitude", ylab="Latitude", main="Pressure standard deviation")
