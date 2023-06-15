"""
Connected Graph
Created on Wed May 31 17:10:28 2023
@author: Jian Wang
"""
import numpy as np
def CG(g, n):
    x=np.array(g)
    value_1 = sum_1 = sum_2 = x
    for i in range(1,n): 
        value_1=np.matmul(value_1, x)
        sum_1=sum_1+value_1
        
    sum_2 = sum_1 + np.identity(n) 
    reachability_matrix = sum_2 > 0.5
    
    if ((reachability_matrix.astype(int)==np.ones((n,n)).astype(int)).all()):      
        return 1 
    else:       
        return 0 
