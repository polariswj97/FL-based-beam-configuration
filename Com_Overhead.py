# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:06:06 2023

@author: Jian Wang
"""
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

from collections import namedtuple
from itertools import count
from PIL import Image
from itertools import combinations

p_g = 0.5
Lambda = 0.001

def Poisson(t, k):
    P_t_k = ((math.exp(-Lambda * t)) * ((Lambda * t) ** k)) / math.factorial(k)
    return P_t_k

def Com_Cost(n, f, p):
    cc = 0
    if f == 0:
        cc = 2 * (n - 1)
    elif f == 1:
        cc = 2 * (n - 1) + p * (n - 1 + 2) + 2 * (1 - p)
    elif f == 2:
        cc = n * (n - 1)
    elif f == 3:
        cc = p * n * (n - 1)
    return cc
    
p_f = (1 - Poisson(200, 0))
print(p_f)

Com_Cost_List = []

for i in range(4):
    cc_list = []
    p = 0
    for j in range(11):
        if i == 1:
            cc_list.append(Com_Cost((j + 2), i, p_f))
        elif i == 3:
            cc_list.append(Com_Cost((j + 2), i, p_g))
        else:
            cc_list.append(Com_Cost((j + 2), i, p))
    
    Com_Cost_List.append(cc_list)
        
plt.figure()
b3 = [i+1 for i in range(len(Com_Cost_List[0]))]
plt.plot(b3, Com_Cost_List[0])
plt.plot(b3, Com_Cost_List[1])
plt.plot(b3, Com_Cost_List[2])
plt.plot(b3, Com_Cost_List[3])
plt.title("Overhead")
plt.xlabel("Com Number")
plt.ylabel("Density") 
plt.show()


txtName = "Overhead_Central.txt"
f = open(txtName, "a+")
for p in range(len(Com_Cost_List[0])):
    new_context = str(Com_Cost_List[0][p]) + '\n'
    f.write(new_context)
f.close()  

txtName = "Overhead_Selection.txt"
f = open(txtName, "a+")
for p in range(len(Com_Cost_List[1])):
    new_context = str(Com_Cost_List[1][p]) + '\n'
    f.write(new_context)
f.close()  

txtName = "Overhead_All.txt"
f = open(txtName, "a+")
for p in range(len(Com_Cost_List[2])):
    new_context = str(Com_Cost_List[2][p]) + '\n'
    f.write(new_context)
f.close()  

txtName = "Overhead_Gossip.txt"
f = open(txtName, "a+")
for p in range(len(Com_Cost_List[3])):
    new_context = str(Com_Cost_List[3][p]) + '\n'
    f.write(new_context)
f.close()  

