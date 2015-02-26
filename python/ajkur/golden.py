from __future__ import division
import numpy as np


# Golden section search in 1D
# Inputs: low bound, high bound, error function, optional print out
# Returns: best value
def goldenSection(it_L,it_H,eCalc,eCalcArgs,verb=False):
    f_calls = 0
    count = 1

    # Low and high error calculations
    eTotal_L = eCalc(it_L, eCalcArgs)
    eTotal_H = eCalc(it_H, eCalcArgs)
    f_calls = f_calls + 2
    if verb == True:
        print("Initial Error Calculations")
        print 'Low Guess: '+str(eTotal_L)
        print 'High Guess: '+str(eTotal_H)
        print("#\tvalue\terror")

    # Golden Ratio
    gold = (1.0 + 5**.5) / 2.0        

    # Calculate first golden point
    it_A = (gold * it_L + it_H)/(gold + 1.0) 
    eTotal_A = eCalc(it_A, eCalcArgs)
    f_calls = f_calls + 1
    if verb == True:
        print "%i\t%.2f\t%.2f" %(count, it_A, eTotal_A)
    count += 1

    while abs(it_L-it_H)>0.05 and count < 50:
        # Calculate next golden point for comparison
        it_B = it_L + it_H - it_A
        eTotal_B = eCalc(it_B, eCalcArgs)
        f_calls = f_calls + 1
        if verb == True:
            print "%i\t%.2f\t%.6f" %(count, it_B, eTotal_B)
            # print it_L,it_A,it_B,it_H
            # print eTotal_L,eTotal_A,eTotal_B,eTotal_H
        count += 1

        # Decide new point assignment based on whether A or B is greater
        if it_A < it_B:
            if eTotal_B>eTotal_A:
                it_H = it_B
                eTotal_H = eTotal_B 
            elif eTotal_B<=eTotal_A:
                it_L = it_A
                eTotal_L = eTotal_A
                it_A = it_B
                eTotal_A = eTotal_B
        elif it_A > it_B:
            if eTotal_B>eTotal_A:
                it_L = it_B
                eTotal_L = eTotal_B 
            elif eTotal_B<=eTotal_A:
                it_H = it_A
                eTotal_H = eTotal_A
                it_A = it_B
                eTotal_A = eTotal_B
    
    return it_A