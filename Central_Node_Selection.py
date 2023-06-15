"""
Created on Tue May 16 14:56:42 2023

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

Lambda = 0.001
T = 100

def Poisson(t, k):
    P_t_k = ((math.exp(-Lambda * t)) * ((Lambda * t) ** k)) / math.factorial(k)
    return P_t_k

Fail1 = []
Fail2 = []
Fail3 = []

for i in range(80):
    Fail1.append(1 - Poisson(i * 100, 0))
    Fail2.append((1 - Poisson(i * 100, 0)) ** 2)
    Fail3.append((1 - Poisson(i * 100, 0)) ** 20)
     

plt.figure()
b3 = [i+1 for i in range(len(Fail1))]
plt.plot(b3, Fail1)
plt.plot(b3, Fail2)
plt.plot(b3, Fail3)
plt.title("Fail Rate")
plt.xlabel("Com Round")
plt.ylabel("Fail Rate") 
plt.show()


txtName = "Fail_Rate_Central.txt"
f = open(txtName, "a+")
for p in range(len(Fail1)):
    new_context = str(Fail1[p]) + '\n'
    f.write(new_context)
f.close()    

txtName = "Fail_Rate_Selection.txt"
f = open(txtName, "a+")
for p in range(len(Fail2)):
    new_context = str(Fail2[p]) + '\n'
    f.write(new_context)
f.close()    

txtName = "Fail_Rate_Discentral.txt"
f = open(txtName, "a+")
for p in range(len(Fail3)):
    new_context = str(Fail3[p]) + '\n'
    f.write(new_context)
f.close()    

