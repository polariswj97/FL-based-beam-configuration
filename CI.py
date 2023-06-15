"""
95% Confidence_Interval
Created on Wed May 24 2023
@author: Jian Wang
"""

import numpy as np
from scipy import stats
 
def CI(data):
    CI_UB = []
    CI_LB = []
    AV = []
    for i in range(len(data[0])):
        vl = []
        for j in range(len(data)):
            vl.append(data[j][i])
            
        avg = np.mean(vl)
        standardError = stats.sem(vl)
        a = avg - 1.96 * standardError
        b = avg + 1.96 * standardError
        AV.append(avg)
        CI_UB.append(b)
        CI_LB.append(a)

    return CI_UB, CI_LB, AV
