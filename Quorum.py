"""
Created on Thu May 19 16:16:30 2022

@author: 31171
"""
#%% 模块导入
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

np.seterr(divide='ignore',invalid='ignore')

#%% 参数设置
"""场景范围"""
Length = 600   #正方形区域，边长500(m),用于产生无人机的初始位置


"""节点/波束数量"""
UAV_Num = 8    #无人机群数量
Beam_Num = 4   #一个微基站可以产生的波束数量上限，每个波束主瓣的宽度为30度
Sec_Num = 12    #微基站扇区数量，由波束宽度为30度可得到
Beam_Wid = (math.pi) / (Sec_Num / 2) #波束宽度

"""功率/增益/频率/带宽/噪声/阈值"""
UAV_Power = 80   #微基站发射功率/dBm
Gain_T = 6      #微基站波束主瓣发射增益
Gain_R =6       #用户端波束接收增益
F_UAV = 28       #毫米波载波频率/GHz
W_UAV = 2        #毫米波基站带宽/GHz 
NF = 4          #噪声功率谱密度
SNR_Thr = -16    #信噪比门限/dB
Dis_Thr = 200    #最大可通信距离


    
#%% 函数定义
"""路径损耗函数"""
def Path_Loss(dis): 
    PL = 32.4 + 17.3 * np.log10(F_UAV) + 20 * np.log10(dis) + 2 * np.random.randn()
    return PL

"""信噪比函数"""
def SNR(N_1, N_2):
    dis = math.sqrt((N_1.location[0] - N_2.location[0]) ** 2 + (N_1.location[1] - N_2.location[1]) ** 2)
    PL = Path_Loss(dis)
    snr = UAV_Power + Gain_T + Gain_R - PL - NF * W_UAV
    return snr

"""数据率函数"""
def Rate(SNR_N1_N2):    
    if SNR_N1_N2 > SNR_Thr:
        SNR_N1_N2 = 10 ** (3 + (SNR_N1_N2 / 10)) #将dB转化为倍数
        rate_u_b = W_UAV * np.log2(1 + SNR_N1_N2) #单位Mps
    else:
        rate_u_b = 0
    return rate_u_b

"""距离扇区计算"""
def Link_Com(N_1, N_2):
    
    N_1_x = N_1.location[0]
    N_1_y = N_1.location[1]
    
    N_2_x = N_2.location[0]
    N_2_y = N_2.location[1]
    
    
    #角度计算,以N_1为坐标原点
    Dis_N1_N2 = math.sqrt((N_1_x - N_2_x) ** 2 + (N_1_y - N_2_y) ** 2)
    Dis_N1_N2_x = N_2_x - N_1_x
    Cos_Val = Dis_N1_N2_x / Dis_N1_N2
    
    if N_2_y >= N_1_y:
        Ang = math.acos(Cos_Val)
    else:
        Ang = 2 * math.pi - math.acos(Cos_Val)
    
    #扇区判断
    sec_Num = Ang / Beam_Wid
    
    if sec_Num == sec_Num:
        sec_Num = math.ceil(sec_Num)
    else:
        sec_Num = 1
      
    return Dis_N1_N2, sec_Num

#%% 类声明
"""
无人机类定义
1.属性：名字(编号)，位置(二维) ，关联波束
2.方法：随机游走；获取名字；获取位置：获取关联波束
"""
class UAV:
    def __init__(self, name):
        #基本参数
        self.name = name  #节点编号
        self.state = 1 #节点工作状态，0为休眠，1为工作
        self.location = np.random.randint(0, 200, 2)  #节点位置
        
        #邻居表以及邻居分布
        self.nei_one = []   #一跳邻居表[邻居编号，扇区]
        self.nei_two = [] #二跳邻居表[邻居编号，扇区]
        self.beam_nei_o = [0 for k in range(Sec_Num)]  #一跳邻居分布扇区
        self.beam_nei_t = [0 for j in range(Sec_Num)]  #二跳邻居分布扇区
        
        self.row_col = [[1,6],[2,5],[3,4],[4,3],[5,2],[6,1],[7,1],[8,2],[9,3],[10,4],[11,5],[12,6]]
        self.sector_slot = [0 for m in range(Sec_Num)]
        
        """新增"""
        self.beam_nei = [0 for g in range(Sec_Num)] #邻居分布
        self.idle_num = 0  #每个时隙错过的邻居
        
        #移动性定义
        self.direction_angle = random.uniform(0, 2 * (math.pi)) #方向角
        self.dis_move = 10 #单位是m,按照速度3km/h,时间间隔6s算出 
        
    def sec_slot_quorum(self, r, c):
        slot_quorum = [0 for i in range(72)]
        row = r - 1
        col = c - 1
        for j in range(6):
            slot_quorum[row * 6 + j] = 1
        for k in range(12):
            slot_quorum[col + k * 6] = 1  
            
        return slot_quorum
    
    def quorum_init(self):
        for r_c in range(len(self.row_col)):
            self.sector_slot[r_c] = self.sec_slot_quorum(self.row_col[r_c][0], self.row_col[r_c][1])
       
    #每个通信周期后，初始化邻居表(由于节点移动)，
    def Par_Init_Class(self):
        self.nei_one = []   #一跳邻居表
        self.nei_two = [] #二跳邻居表
        
        self.beam_nei_o = [0 for k in range(Sec_Num)]  #一跳邻居分布扇区
        self.beam_nei_t = [0 for j in range(Sec_Num)]  #二跳邻居分布扇区
        self.beam_nei = [0 for g in range(Sec_Num)] #邻居分布 
        
    #更新邻居扇区分布，计算奖励以及下一个状态
    def Cal_Beam_Nei(self):       
        self.beam_nei_o = [0 for k in range(Sec_Num)]  
        self.beam_nei_t = [0 for j in range(Sec_Num)]
        
        for nei_one_i in self.nei_one:
            self.beam_nei_o[nei_one_i[1] - 1] = self.beam_nei_o[nei_one_i[1] - 1] + 1
        for nei_two_i in self.nei_two:
            self.beam_nei_t[nei_two_i[1] - 1] = self.beam_nei_t[nei_two_i[1] - 1] + 1
               
    def act(self, slot_num): 
        act_q = []
        index_q = slot_num % 72
        for pp in range(Sec_Num):
            if (self.sector_slot[pp][index_q] == 1):
                act_q.append(pp + 1)
        return act_q
    
    def move(self):   #用户移动函数        
        self.location[0] = self.location[0] + self.dis_move * math.cos(self.direction_angle)
        self.location[1] = self.location[1] + self.dis_move * math.sin(self.direction_angle)
        
        self.direction_angle = self.direction_angle + random.uniform(-((math.pi) / 4) , ((math.pi) / 4))
        if self.direction_angle < 0:
            self.direction_angle = self.direction_angle + 2 * (math.pi)
            
        """越界返回"""
        if self.location[0] > Length:
            self.location[0] -= 50
        elif self.location[0] < 0:
            self.location[0] += 50 
            
        if self.location[1] > Length:
            self.location[1] -= 50
        elif self.location[1] < 0:
            self.location[1] += 50

#%% 主程序部分
Comm_Num = 100 #通信周期数量
Slot_Num  = 100 #每个通信周期邻居发现阶段的时隙数量

UAV_NNet_Weight_List = [0 for p in range(UAV_Num)]        #所有节点的神经网络权重参数集合
UAV_List = [] #无人机群集合

for e in range(UAV_Num): #创建无人机群
    UAV_List.append(UAV(e+1))
    UAV_List[e].quorum_init()
    
def Location_Set(Loc_List,infile):
    f = open(infile,'r')
    sourceInLine = f.readlines()
    Loc_Set = []
    
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split(' ')
        Loc_Set.append(temp2)
    for i in range(len(Loc_Set)):
        for j in range(2):
            Loc_Set[i].append(float(Loc_Set[i][j]))
        del(Loc_Set[i][0:2])
        
    flag = 0
    for lis in Loc_List:
        lis.location[0] = Loc_Set[flag][0]
        lis.location[1] = Loc_Set[flag][1]
        flag = flag + 1

def Connected_Graph(g, n):
    x=np.array(g)#利用numpy库将输入的列表转换为numpy中矩阵
    value_1 = sum_1 = sum_2 = x
    for i in range(1,n): #计算可达矩阵
        value_1=np.matmul(value_1, x)
        sum_1=sum_1+value_1
        
    sum_2 = sum_1 + np.identity(n) 
    reachability_matrix = sum_2 > 0.5
    if ((reachability_matrix.astype(int)==np.ones((n,n)).astype(int)).all()):
        #print("此图为连通图")
        return 1
    else:
        #print("此图不连通")
        return 0          

#infile_1 = 'Node_Loc_8.txt'
#Location_Set(UAV_List,infile_1)

'''所有可能的链路'''
Node_list = [(i+1) for i in range(UAV_Num)]  
Link_Space = list(combinations(Node_list, 2))

LP_List = [0 for i in range(Comm_Num)]
#学习主程序
for com_r in range(Comm_Num):
    print("Comm_Round",com_r + 1)
    Link_Set = [] #所有可建立的链路
    Link_Dis = [] #链路长度
    Link_Sec = [] #链路两端节点对应扇区
    Link_Set_Exi = [] #已建立连接的链路
    Link_percent = [0 for i in range(Slot_Num)] #已建立的链路比例 
    
    Link_matrix = [[0 for i in range(UAV_Num)] for j in range(UAV_Num)] #邻接矩阵
    
    for link_i in Link_Space:
        dis_i, sec_i = Link_Com(UAV_List[link_i[0]-1], UAV_List[link_i[1]-1])
        if dis_i < Dis_Thr:
            Link_Set.append(link_i)
            Link_Dis.append(dis_i)
            if sec_i > (Sec_Num / 2):
                sec_i_o = int(sec_i - (Sec_Num / 2))
            else:
                sec_i_o = int(sec_i + (Sec_Num / 2))
            Link_Sec.append([sec_i, sec_i_o]) 
    
    """新增"""
    for link_k in range(len(Link_Set)):
        node_a = Link_Set[link_k][0] - 1
        node_b = Link_Set[link_k][1] - 1
        
        #统计每个节点的每个扇区邻居数量，为计算碰撞概率做准备
        
        UAV_List[node_a].beam_nei[Link_Sec[link_k][0] - 1] += 1
        UAV_List[node_b].beam_nei[Link_Sec[link_k][1] - 1] += 1

    for uav_i in range(UAV_Num):
            UAV_List[uav_i].move()       

    for slot_i in range(Slot_Num): 
        """新增"""
        Act_all_node = [] #所有节点在本时隙内的动作(波束指向)，为了检测波束对齐做准备
        
        #计算一跳邻居
        for uav_j in range(UAV_Num):
            UAV_List[uav_j].idle_num = 0 
            act_uav_j = UAV_List[uav_j].act(slot_i)
            
            """新增"""
            Act_all_node.append(act_uav_j)
        
        """新增"""            
        for link_j in range(len(Link_Set)):
            
            """新增"""
            node_c = Link_Set[link_j][0] - 1
            node_d = Link_Set[link_j][1] - 1
            
            if (Link_Sec[link_j][0] in Act_all_node[node_c]) and (Link_Sec[link_j][1] in Act_all_node[node_d]): #波束对齐
                nei_num_1 = UAV_List[node_c].beam_nei[Link_Sec[link_j][0] - 1] 
                nei_num_2 = UAV_List[node_d].beam_nei[Link_Sec[link_j][1] - 1]

                   
                nei_num_sum = (nei_num_1 + nei_num_2)
                
                if nei_num_sum > 10:
                    nei_num_sum = 10
                elif nei_num_sum < 2:
                    nei_num_sum = 2
                    
                col_pro = (2 ** nei_num_sum - 4) / (2 ** 10 + 200) #碰撞概率
                
                #满足条件则修改邻居表
                if random.random() > col_pro:
                    #print("check")
                    if UAV_List[node_c].beam_nei[Link_Sec[link_j][0] - 1] > 0:
                        UAV_List[node_c].beam_nei[Link_Sec[link_j][0] - 1] -= 1
                    if UAV_List[node_d].beam_nei[Link_Sec[link_j][1] - 1] > 0:
                        UAV_List[node_d].beam_nei[Link_Sec[link_j][1] - 1] -= 1
                    if Link_Set[link_j] not in Link_Set_Exi:
                        Link_Set_Exi.append(Link_Set[link_j])
                    
                    #修改一跳邻居表
                    if [Link_Set[link_j][1],Link_Sec[link_j][0]] not in UAV_List[node_c].nei_one:
                        if [Link_Set[link_j][1],Link_Sec[link_j][0]] in UAV_List[node_c].nei_two:
                            UAV_List[node_c].nei_two.remove([Link_Set[link_j][1],Link_Sec[link_j][0]])  
                        Link_matrix[node_c][node_d] = 1
                        UAV_List[node_c].nei_one.append([Link_Set[link_j][1],Link_Sec[link_j][0]])
                    
                    if [node_c + 1,Link_Sec[link_j][1]] not in UAV_List[node_d].nei_one:
                        if [node_c + 1,Link_Sec[link_j][1]] in UAV_List[node_d].nei_two:
                            UAV_List[node_d].nei_two.remove([node_c + 1,Link_Sec[link_j][1]])
                        Link_matrix[node_d][node_c] = 1
                        UAV_List[node_d].nei_one.append([node_c + 1,Link_Sec[link_j][1]])


        #更新节点扇区-邻居分布数据,计算状态、奖励等数据并保存
        for uav_i in range(UAV_Num):
            UAV_List[uav_i].Cal_Beam_Nei()
        
        if Connected_Graph(Link_matrix, UAV_Num) == 1:
            print("连通图形成",slot_i)                     
      
        link_percent_i = (len(Link_Set_Exi)/len(Link_Set)) * 100
        Link_percent[slot_i] = link_percent_i
        print("Slot_Num", slot_i + 1, link_percent_i)
    
    LP_List[com_r] = Link_percent
                        
   
    plt.figure()
    b5 = [i+1 for i in range(len(Link_percent))]
    plt.plot(b5,Link_percent,"-ok")
    plt.title("Link")
    plt.xlabel("slot num")
    plt.ylabel("Link percent") 
    plt.show()


Link_percent = [0 for i in range(Slot_Num)]
for i in range(Slot_Num):
    tem = 0
    for j in range(Comm_Num):
        tem = tem + LP_List[j][i]
    Link_percent[i] = tem / Comm_Num
        
    
txtName = "Link_Percent_Quorum.txt"
f = open(txtName, "a+")
for k in range(len(Link_percent)):
    new_context = str(Link_percent[k]) + '\n'
    f.write(new_context)
f.close()           

