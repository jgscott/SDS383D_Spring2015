from __future__ import division
import numpy as np  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os

# Data generating function for testing out kernal smoothing
class dataGen:

    # Arbitrary data generator for a 4th order function
    # Inputs: number of data points, plotting(T/F)
    # Returns: x data, y data
    def fourGen(self,n=500,plotting=False):
        # Draw n x points on an arbitrary interval
        xD = np.random.uniform(-3,3,n)

        # Draw n errors from a normal with arbitrary variance
        eD = np.random.normal(scale=5,size=n)

        # Generate y points with an arbitrary function
        yD = xD**4+2*xD**3-3*xD**2+5+eD

        # Optional plotting
        if plotting == True:
            xP = np.linspace(-3,3,100)
            yP = xP**4+2*xP**3-3*xP**2+5
            plt.figure()
            plt.plot(xD,yD,'bo',alpha=0.5)
            plt.plot(xP,yP,'r-',lw=2)
            plt.title('Original function and noisy data')
            plt.savefig('./Figures/Original_and_Data.png',bbox_inches='tight')
            plt.close()

        return xD,yD

    # Arbitrary data generator for testing the cross validation problem
    # Wiggly function and smooth function on the unit interval with high and low noise
    # Inputs: function type (Wiggle/Smooth), noise variance, number of data points, plotting(T/F)
    # Returns: x data, y data
    def crossGen(self,funType,var,n=500,plotting=False):
        # Draw n x points on an arbitrary interval
        xD = np.random.uniform(0,1,n)

        # Draw n errors from a normal with arbitrary variance
        eD = np.random.normal(scale=var,size=n)

        # Generate y points with an arbitrary function
        if funType == 'Smooth':
            yD = np.exp(xD)+1.0+eD
        elif funType == 'Wiggle':
            yD = np.sin(6.0*xD)+0.2*np.sin(24.0*xD)+0.3*np.cos(40.0*xD)+eD

        # Optional plotting
        if plotting == True:
            xP = np.linspace(0,1,100)
            if funType == 'Smooth':
                yP = np.exp(xP)+1.0
            elif funType == 'Wiggle':
                yP = np.sin(6.0*xP)+0.2*np.sin(24.0*xP)+0.3*np.cos(40.0*xP)
            plt.figure()
            plt.plot(xD,yD,'ro',alpha=0.5)
            plt.plot(xP,yP,'b-',lw=2)
            plt.title(funType+' function and noisy data of variance '+str(var))
            plt.savefig('./Figures/CV_'+funType+'_and_Data_'+str(var)+'.png',bbox_inches='tight')
            plt.close()

        return xD,yD


# Container for kernal smoothing functions
# There are four main sections here:
#   Kernal Regression
#   Local Linear Regression
#   Cross Validation
#   Kernals
class kernSmooth:    
    def __init__(self,kernName='Gaussian'):
        # Set kernal to Gaussian by default
        self.setKern(kernName)

        # Check for figures directory and make one if necessary
        if os.path.isdir('./Figures') == True: pass
        else: os.mkdir('./Figures')


    #########################
    ### Kernal Regression ###
    #########################

    # Kernal Regression
    # Inputs: x data, y data, x star, bandwidth 
    # Returns: y star
    def kernReg(self,xD,yD,xS,h):
        # Calculate weights for x star
        wD = self.calcWeight(xD,xS,h)

        # Compute new y star from weights
        yS = 0.0
        for i in range(xD.shape[0]):
            yS += wD[i]*yD[i]

        return yS            

    # Weight calculation function
    # Inputs: x data, x star, bandwidth, normalization (T/F)
    # Returns: (normalized) weights
    def calcWeight(self,xD,xS,h,norm=True):
        # Normalizing constant
        wN = 0.0

        # Weights
        wD = np.zeros(xD.shape[0])
        for i in range(xD.shape[0]):
            wD[i] = self.kern((xD[i]-xS)/h)/h
            wN += wD[i]

        # Normalize weights
        if norm == True:
            wD = wD/wN

        return wD

    # Wrapper for kernal regression that plots the predictions over the data range
    # Inputs: training data, bandwidth, plot title
    def kernPlot(self,xD,yD,h,plTitle='None'):
        # Find range of x values
        xMin = np.min(xD)
        xMax = np.max(xD)

        # Generate array of new x values and empty y array for plotting the curve fit
        nS = 100
        xS = np.linspace(xMin,xMax,nS)
        yS = np.zeros(nS)

        # Loop through each new point x star and calculate y star
        for j in range(nS):
            yS[j] = self.kernReg(xD,yD,xS[j],h)

        # Plot data and prediction
        plt.figure()
        plt.plot(xD,yD,'bo',alpha=0.5)
        plt.plot(xS,yS,'k-',lw=2)
        plt.xlim([xMin,xMax])
        plt.title(plTitle+', h='+str(h))
        plt.savefig('./Figures/SM_'+plTitle+'_'+str(h)+'.png',bbox_inches='tight')
        plt.close()


    ###############################
    ### Local Linear Regression ###
    ###############################

    # Local linear regression
    # Inputs: x data, y data, x star, bandwidth, loocv output (T/F), index of loocv point
    # Returns: y star, loocv weight
    def localLinear(self,xD,yD,xS,h,loo=False,looI=0):
        # Calculate weights for x star
        wD = self.calcWeight(xD,xS,h,norm=False)

        # Calculate s_0, s_1, and s_2
        s_j = np.zeros(3)
        for j in range(3):
            s_j[j] = self.sCalc(xD,xS,wD,j)

        # Calculate estimate: y star
        yS = 0.0
        hi = np.zeros(xD.shape[0])
        for i in range(xD.shape[0]):
            yS += (s_j[2]-s_j[1]*(xD[i]-xS))*wD[i]*yD[i]
            hi[i] = (s_j[2]-s_j[1]*(xD[i]-xS))*wD[i]/(s_j[0]*s_j[2]-s_j[1]**2)

        yS = yS/(s_j[0]*s_j[2]-s_j[1]**2)

        return yS,hi

    # Calculate 's' term for local regression
    # Inputs: x data, x star, weights, power
    # Return: s_j
    def sCalc(self,xD,xS,wD,j):
        s_j = 0.0
        for i in range(xD.shape[0]):
            s_j += wD[i]*(xD[i]-xS)**j
        return s_j

    # Calculate residuals at each data point and smoothing matrix
    # Inputs: x data, y data, bandwidth
    # Returns: residuals, smoothing matrix
    def resWeight(self,xD,yD,h):
        res = np.zeros(yD.shape[0])
        HH = np.zeros([yD.shape[0],yD.shape[0]])
        for i in range(yD.shape[0]):
            yHat, HH[i,:] = self.localLinear(xD,yD,xD[i],h)
            res[i] = yD[i] - yHat

        return res, HH

    # Confidence interval estimation
    # Inputs: residuals, smoothing matrix
    # Returns: 95 percent confidence interval
    def ciEst(self,res,HH):
        # Estimate std dev        
        rss = 0.0
        for i in range(res.shape[0]):
            rss += res[i]**2
        sigSqHat = rss/(res.shape[0]-2.0*np.trace(HH)+np.trace(np.dot(HH.T,HH)))

        # Estimate 95 percent confidence intervals
        CI = np.zeros(res.shape[0])
        for i in range(res.shape[0]):
            CI[i] = 1.96*(sigSqHat*HH[i,i])**0.5
            # CI[i] = (sigSqHat*np.sum(HH[i,:]*HH[i,:]))**0.5

        return CI

    # Wrapper for local linear regression that plots the predictions over the data range
    # Inputs: training data, bandwidth, plot title
    def localLinearPlot(self,xD,yD,h,plTitle='None',xLab='',yLab='',CIflag=True,logPlot=False):
        # Find range of x values
        xMin = np.min(xD)
        xMax = np.max(xD)

        # Generate array of new x values and empty y array for plotting the curve fit
        nS = 100
        xS = np.linspace(xMin,xMax,nS)
        yS = np.zeros(nS)

        # Loop through each new point x star and calculate y star
        for j in range(nS):
            yS[j], hi = self.localLinear(xD,yD,xS[j],h)

        # Calculate residuals and smoothing matrix
        res, HH = self.resWeight(xD,yD,h)

        # Calculate 95 percent confidence interval
        CI = self.ciEst(res,HH)

        # Sort xD, yD, res, and CI for line plotting
        inds = xD.argsort()
        xD_sort = xD[inds]
        yD_sort = yD[inds]
        res_sort = res[inds]
        CI_sort = CI[inds]

        # Plot data and prediction
        plt.figure()
        if logPlot == False:
            plt.plot(xD,yD,'bo',alpha=0.5)
            plt.plot(xS,yS,'k-',lw=2)
            plt.ylabel(yLab)
        else:
            plt.plot(xD,np.log(yD),'bo',alpha=0.5)
            plt.plot(xS,np.log(yS),'k-',lw=2)
            plt.ylabel('Log '+yLab)
        if CIflag == True:
            plt.plot(xD_sort,yD_sort-res_sort+CI_sort,'k--',lw=2)
            plt.plot(xD_sort,yD_sort-res_sort-CI_sort,'k--',lw=2)
        plt.xlim([xMin,xMax])
        plt.xlabel(xLab)
        plt.title(plTitle+', h='+str(h))
        plt.savefig('./Figures/LL_'+plTitle+'.png',bbox_inches='tight')
        plt.close()        

        # Plot residuals
        plt.figure()
        if logPlot == False:
            plt.plot(xD,res,'bo',alpha=0.7)
            plt.ylabel('Residual '+yLab)
        else:
            plt.plot(xD,np.log(yD)-np.log(yD-res),'bo',alpha=0.7)
            plt.ylabel('Log Residual '+yLab)
        plt.plot(xD,np.zeros(xD.shape[0]),'k-',lw=2)
        if CIflag == True:
            plt.plot(xD_sort,np.zeros(xD.shape[0])+CI_sort,'k--',lw=2)
            plt.plot(xD_sort,np.zeros(xD.shape[0])-CI_sort,'k--',lw=2)
        plt.xlim([xMin,xMax])
        plt.xlabel(xLab)
        plt.title(plTitle+', h='+str(h))
        plt.savefig('./Figures/LL_'+plTitle+'_residual.png',bbox_inches='tight')
        plt.close()


    ########################
    ### Cross Validation ###
    ########################

    # Leave one out cross validation for local linear smoother
    # Inputs: x data, y data, bandwidth
    # Return: LOOCV metric
    def loocv(self,xD,yD,h):        
        err = 0.0
        for i in range(yD.shape[0]):
            yHat, hi = self.localLinear(xD,yD,xD[i],h)
            err += ((yD[i]-yHat)/(1.0-hi[i]))**2

        return err

    # Wrapper for minimization of bandwidth with loocv
    # Inputs: bandwidth, (x data, y data)
    # Return: LOOCV metric
    def loocvWrap(self,h,(xD,yD)):
        return self.loocv(xD,yD,h)

    # Cross Validation
    # Inputs: training data, test data, bandwidth list, plotting (T/F), plot title
    # Returns: mean squared error
    def crossVal(self,xD_train,yD_train,xD_test,yD_test,h,plotting=False,plTitle='None'):
        # Mean square error
        mse = np.zeros(len(h))

        # Compute prediction for each h
        for i in range(len(h)):

            # Array of predicted y values
            yP = np.zeros(yD_test.shape[0])

            # Loop through each new point x test
            for j in range(xD_test.shape[0]):
                yP[j] = self.kernReg(xD_train,yD_train,xD_test[j],h[i])

                # Add difference of prediction and test squared to MSE
                mse[i] += (yP[j]-yD_test[j])**2

            # Take mean of squared errors
            mse[i] = mse[i]/yP.shape[0]

            # Optional plotting of test set and prediction
            if plotting == True:
                plt.figure()
                plt.plot(xD_test,yD_test,'ro',alpha=0.5)
                # Sort xD_test for line plotting
                inds = xD_test.argsort()
                xD_sort = xD_test[inds]
                yP_sort = yP[inds]
                plt.plot(xD_sort,yP_sort,'k-',lw=2)
                plt.title('Cross Validation: '+plTitle+', h='+str(h[i])+', MSE='+str(np.round(mse[i],2)))
                plt.savefig('./Figures/CV_'+plTitle+'_'+str(h[i])+'.png',bbox_inches='tight')
                plt.close()
        return mse

    # Wrapper for Cross Validation h minimization
    # Inputs: exponential value, training and test data as a tuple
    # Returns: mean squared error
    def cvWrap(self,expVal,(xD_train,yD_train,xD_test,yD_test)):
        err = self.crossVal(xD_train,yD_train,xD_test,yD_test,[np.exp(expVal)])[0]
        return err


    ###############
    ### Kernals ###
    ###############

    # Kernal setting function
    # Input: kernal name
    #
    #   Valid Kernals:
    #       Gaussian
    #       Uniform
    def setKern(self,kernName):
        if kernName == 'Gaussian':
            self.kern = self.gausKern
        elif kernName == 'Uniform':
            self.kern = self.uniKern
        else:
            print 'Please specify a valid kernal name!'

    # Uniform kernal function
    # Input: x
    # Returns: uniform kernal at x
    def uniKern(self,x):
        if abs(x) <= 1:
            kF = 0.5
        else:
            kF = 0.0
        return kF

    # Gaussian kernal function
    # Input: x
    # Returns: gaussian kernal at x
    def gausKern(self,x):
        return np.exp(-0.5*x**2)/(2.0*np.pi)**0.5
