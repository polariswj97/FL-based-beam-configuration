"""
Created on May 2 2023
@author: Jian Wang
"""
import DBC
import FLBC
import PFL


UAV_Num_List = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
Sec_Num_List = [8, 9, 12, 18, 24, 36]
Beam_Num_List = [2, 3, 4, 5, 6]
LR_Rate_List = [0.001, 0.003, 0.01, 0.03]

for i in range(10):
    for j in range(6):
        for k in range(5):
            for l in range(100):
                FLBC.Par_Set(UAV_Num_List[i], Sec_Num_List[j], Beam_Num_List[k])
                FLBC.FLBC()
                
                DBC.Par_Set(UAV_Num_List[i], Sec_Num_List[j], Beam_Num_List[k])
                DBC.DBC()
                
                PFL.Par_Set(UAV_Num_List[i], Sec_Num_List[j], Beam_Num_List[k])
                PFL.PFL()



