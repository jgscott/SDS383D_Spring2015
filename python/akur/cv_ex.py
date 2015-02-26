from __future__ import division
import numpy as np 
import kernal_smooth
import sys
import dhsimp


### Cross Validation Exercises ###

# Make class objects for kernal smoothing functions and data generators
ks = kernal_smooth.kernSmooth()
dg = kernal_smooth.dataGen()

# Part A: Generate test and training data sets for testing the cross validation fucntion
xD_train, yD_train = dg.fourGen()
xD_test, yD_test = dg.fourGen()

# Set gaussian kernal
ks.setKern('Gaussian')
print ks.crossVal(xD_train,yD_train,xD_test,yD_test,[0.01,0.05,0.1,0.5],plotting=True,plTitle='Gaussian')

# Set uniform kernal
ks.setKern('Uniform')
print ks.crossVal(xD_train,yD_train,xD_test,yD_test,[0.01,0.05,0.1,0.5],plotting=True,plTitle='Uniform')

# Part B
funType = ['Wiggle','Smooth']
varList = ['0.1','0.5']
for i in range(2):
    for j in range(2): 
        # Generate test and training data sets for the four functions in Part B
        xD_train, yD_train = dg.crossGen(funType[i],varList[j],n=500,plotting=True)
        xD_test, yD_test = dg.crossGen(funType[i],varList[j],n=500,plotting=True)

        # Minimize mse by choosing h with downhill simplex
        arTest = np.zeros(1)-0.4
        iCur,myPathX,myPathY,bestAr = dhsimp.dhSimp(arTest,ks.cvWrap,(xD_train,yD_train,xD_test,yD_test),prOut=True)  

        # Plot best value
        ks.crossVal(xD_train,yD_train,xD_test,yD_test,[np.exp(bestAr[0])],plotting=True,plTitle=funType[i]+'Best'+str(j))
