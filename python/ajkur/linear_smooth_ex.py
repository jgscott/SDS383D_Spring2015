from __future__ import division
import numpy as np 
import kernal_smooth


### Linear Smoothing Exercises ###

# Make class objects for kernal smoothing functions and data generators
ks = kernal_smooth.kernSmooth()
dg = kernal_smooth.dataGen()

# Generate some data
xD,yD = dg.fourGen(plotting=True)

# Subtract the means
xD = xD - np.mean(xD)
yD = yD - np.mean(yD)

# Kernal regression on data for a few different h values and kernals

# Set uniform kernal
ks.setKern('Uniform')

# Uniform low h
ks.kernPlot(xD,yD,0.1,plTitle='Uniform')

# Uniform mid h
ks.kernPlot(xD,yD,0.5,plTitle='Uniform')

# Uniform high h
ks.kernPlot(xD,yD,1.0,plTitle='Uniform')

# Set gaussian kernal
ks.setKern('Gaussian')

# Gaussian low h
ks.kernPlot(xD,yD,0.1,plTitle='Gaussian')

# Gaussian good h
ks.kernPlot(xD,yD,0.2,plTitle='Gaussian')

# Gaussian mid h
ks.kernPlot(xD,yD,0.5,plTitle='Gaussian')

# Gaussian high h
ks.kernPlot(xD,yD,1.0,plTitle='Gaussian')
