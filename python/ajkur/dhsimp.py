from __future__ import division
import numpy as np 
import copy

# Calculate centroid
def centCalc(xVec):
    xC = np.zeros(xVec.shape[0])
    for i in range(xVec.shape[0]):
        for j in range(xVec.shape[1]):
            xC[i] = xC[i] + xVec[i,j]
    return xC/xVec.shape[1]

# Downhill Simplex
def dhSimp(x0,theFun,funArgs,alpha=1.0,gamma=2.0,rho=-0.5,sigma=0.5,tol=1e-8,prOut=True):
    if prOut == True:
        print '\nStarting Downhill Simplex'
        print 'Sum of\tStd dev of\tIteration'
        print 'Errors\tErrors'
    # Store path as a list
    myPathX = []
    myPathY = []
    myPathX.append(x0[0])
    # myPathY.append(x0[1])

    # Get dimensions
    N = x0.shape[0]

    # Error values array
    # errs = np.zeros(N+1)

    # Set vertices of a regular simplex
    verts = np.zeros([N,N+1])
    verts1 = np.zeros([N,N])
    errs = np.arange(N+1,dtype=float)
    errs1 = np.zeros(N)
    verts[0,0] = 1.0
    for i in range(N):
        verts[0,i+1] = -1.0/N
    for k in range(1,N):
        vTemp = 0.0
        for j in range(0,k):
            vTemp = vTemp + verts[j,k]**2
        verts[k,k] = (1.0-vTemp)**0.5
        for i in range(k+1,N+1):
            vTemp = 0.0
            for l in range(0,k):
                vTemp = vTemp + verts[l,k]*verts[l,i]
            verts[k,i] = (-1.0/N - vTemp)/verts[k,k]

    # Shift simplex to specified starting centroid
    for i in range(N):
        for j in range(N+1):
            verts[i,j] = verts[i,j] + x0[i]

    # print verts,np.exp(verts)
    iCur = 0
    while np.sum(errs) > tol and np.std(errs) > tol and iCur < 1000:
        # Calculate erros at each vertex
        for i in range(N+1):
            errs[i] = theFun(verts[:,i],funArgs)
        N1Loc = np.argmax(errs)
        OneLoc = np.argmin(errs)
        xN1 = verts[:,N1Loc]
        xOne = verts[:,OneLoc]
        if iCur%2 == 0 and prOut == True:
            print str(np.sum(errs))+'\t'+str(np.std(errs))+'\t'+str(iCur)
        # Cut out max error
        j = 0
        for i in range(N+1):
            if i != N1Loc:
                verts1[:,j] = verts[:,i]
                errs1[j] = errs[i]
                j += 1

        secLoc = np.argmax(errs1)
        xN = verts1[:,secLoc]

        # Calculate new centroid
        x0 = centCalc(verts1)
        myPathX.append(x0[0])
        # myPathY.append(x0[1])

        # Reflection
        xR = x0 + alpha*(x0 - xN1)
        errR = theFun(xR,funArgs)
        if errR < errs1[secLoc] and errR > errs[OneLoc]:
            verts[:,N1Loc] = 1.0*xR

        # Expansion
        elif errR < errs[OneLoc]:
            xE = x0 + gamma*(x0 - xN1)
            errE = theFun(xE,funArgs)
            if errE < errR:
                verts[:,N1Loc] = 1.0*xE
            else:
                verts[:,N1Loc] = 1.0*xR

        # Contractiion
        else:
            xC = x0 + rho*(x0 - xN1)
            errC = theFun(xC,funArgs)
            if errC < errs[N1Loc]:
                verts[:,N1Loc] = 1.0*xC
            else:
                # Reduction
                for ll in range(N+1):
                    if ll != OneLoc:
                        verts[:,ll] = xOne + sigma*(verts[:,ll] - xOne)
        iCur += 1
    x0 = centCalc(verts)
    myPathX.append(x0[0])
    # myPathY.append(x0[1])
    if prOut == True:
        print '\nComplete\n# of iterations:',iCur
        print 'Error at best point:',theFun(x0,funArgs)
        print 'Best Point:',x0
    return iCur,myPathX,myPathY,x0