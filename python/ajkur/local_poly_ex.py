# from __future__ import division
import numpy as np 
import kernal_smooth
import golden


### Local Polynomial Regression Exercises ###

# Make class object for kernal smoothing functions
ks = kernal_smooth.kernSmooth(kernName='Gaussian')

# Read in utilities data and set y = dollars/days and x = average temperature
data = np.genfromtxt('./utilities.csv',delimiter=',',names=True)
xD = data['temp']
yD = data['gasbill']/data['billingdays']

# Test local linear regression function
ks.localLinearPlot(xD,yD,1.0,plTitle='Test_1',xLab='Temperature (F)',yLab='Cost (Dollars/Day)',CIflag=False)
ks.localLinearPlot(xD,yD,5.0,plTitle='Test_5',xLab='Temperature (F)',yLab='Cost (Dollars/Day)',CIflag=False)

# Leave one out cross validation to choose the best h
besth = golden.goldenSection(1.0,10.0,ks.loocvWrap,(xD,yD),verb=True)
ks.localLinearPlot(xD,yD,besth,plTitle='best_h',xLab='Temperature (F)',yLab='Cost (Dollars/Day)',CIflag=False)

# Log plots
ks.localLinearPlot(xD,yD,besth,plTitle='best_h_log',xLab='Temperature (F)',yLab='Cost (Dollars/Day)',CIflag=False,logPlot=True)

# Plotting confidence intervals
ks.localLinearPlot(xD,yD,besth,plTitle='best_h_CI',xLab='Temperature (F)',yLab='Cost (Dollars/Day)')
# ks.localLinearPlot(xD,yD,besth,plTitle='best_h_log_CI',xLab='Temperature (F)',yLab='Cost (Dollars/Day)',logPlot=True)

