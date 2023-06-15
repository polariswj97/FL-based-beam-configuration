"""
Created on Tue Mar 15 2022

Beam control based on FL+MAML

@author: Jian Wang

"""
import CG
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os

from collections import namedtuple
from itertools import count
from PIL import Image
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


Length = 200
Range = 800
UAV_Num = 10   
Beam_Num = 3    
Sec_Num = 12    
Beam_Wid = (math.pi) / (Sec_Num / 2) 
UAV_Power = 80   
Gain_T = 6     
Gain_R =6       
F_UAV = 28       
W_UAV = 2       
NF = 4          
SNR_Thr = -16    
Dis_Thr = 300   
Pool_Capacity = 1000
LR_MAML = 0.003    
LR_Actor = 0.003    
LR_Critic = 0.003   
LR_Global = 1   
Tau = 0.05          
GAMMA = 0.95         
Sigma = 0.001   
Rho_1 = 0.5        
Rho_2 = 0.5         
Beta_1 = 500        
Beta_2 = 60         
Beta_3 = 60         
State_weight_1 = 90
State_weight_2 = 30
State_weight_3 = 10
Batch_Size = 16     
NNet_Layer_Num = 4  
K = 6 
State_Dim = Sec_Num 
Action_Dim = Beam_Num 
Sec_list = [(i+1) for i in range(Sec_Num)] 
Act_Space = list(combinations(Sec_list, Beam_Num)) 

def Par_Set(u, s, b):
    global UAV_Num
    global Sec_Num
    global Beam_Num 
    global Beam_Wid 
  
    global State_Dim 
    global Action_Dim
    
    global Sec_list
    global Act_Space
    
    UAV_Num = u
    Sec_Num = s
    Beam_Num = b 
    Beam_Wid = (math.pi) / (s/ 2) 
  
    State_Dim = s 
    Action_Dim = b
    Sec_list = [(i+1) for i in range(Sec_Num)]  
    Act_Space = list(combinations(Sec_list, Beam_Num))


def Path_Loss(dis): 
    PL = 32.4 + 17.3 * np.log10(F_UAV) + 20 * np.log10(dis) # + 2 * np.random.randn()
    return PL

def SNR(N_1, N_2):
    dis = math.sqrt((N_1.location[0] - N_2.location[0]) ** 2 + (N_1.location[1] - N_2.location[1]) ** 2)
    PL = Path_Loss(dis)
    snr = UAV_Power + Gain_T + Gain_R - PL - NF * W_UAV
    return snr

def Rate(SNR_N1_N2):    
    if SNR_N1_N2 > SNR_Thr:
        SNR_N1_N2 = 10 ** (3 + (SNR_N1_N2 / 10))
        rate_u_b = W_UAV * np.log2(1 + SNR_N1_N2) 
    else:
        rate_u_b = 0
    return rate_u_b

def Link_Com(N_1, N_2):
    
    N_1_x = N_1.location[0]
    N_1_y = N_1.location[1]
    
    N_2_x = N_2.location[0]
    N_2_y = N_2.location[1]
    

    Dis_N1_N2 = math.sqrt((N_1_x - N_2_x) ** 2 + (N_1_y - N_2_y) ** 2)
    Dis_N1_N2_x = N_2_x - N_1_x
    Cos_Val = Dis_N1_N2_x / Dis_N1_N2
    
    if N_2_y >= N_1_y:
        Ang = math.acos(Cos_Val)
    else:
        Ang = 2 * math.pi - math.acos(Cos_Val)

    sec_num = Ang / Beam_Wid
    
    if sec_num == sec_num:
        sec_num = math.ceil(sec_num)
    else:
        sec_num = 1
      
    return Dis_N1_N2, sec_num


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity #列表容量
        self.memory = [] #列表数据
        
    def push(self, *args): #放入数据
        if len(self.memory) == self.capacity:
            self.memory.pop(0) #列表已满，删除最初放进去的数据
        self.memory.append(args)
           
    def sample(self, batch_size): #采样数据
        return random.sample(self.memory, batch_size)
    
    def data_num(self): #数据量
        return len(self.memory)
    


class Actor(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

        
    def forward(self, s):
        x = F.relu(self.layer1(s))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

 
class Critic(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)       
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)        
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)       
        self.layer4 = nn.Linear(n_hidden_3, out_dim)
        
    def forward(self, s, a, m):
        s = s.float()
        a = a.float()
        x = torch.cat([s, a], m) 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x    
    


class UAV:
    def __init__(self, name):
        self.name = name 
        self.state = 1 
        self.location = np.random.randint(0, 200, 2) 
        
        #邻居表以及邻居分布
        self.nei_one = []   
        self.nei_two = []   
        
        self.nei_beam_one = [0 for n in range(Sec_Num)]
        
        self.beam_nei_o = [0 for k in range(Sec_Num)]  
        self.beam_nei_t = [0 for j in range(Sec_Num)]  
        
      
        self.beam_nei = [0 for g in range(Sec_Num)] 
        self.col_num = [0 for h in range(Sec_Num)] 
        
        self.slot_num = [0] 
        
        self.reward = 0 #
        self.action = [0 for w in range(Beam_Num)]
        self.state_pre = [2 for p in range(State_Dim)] 
        self.state_nex = [2 for m in range(State_Dim)]
        
        self.action_set = []
        self.reward_set = [] 
        self.Loss_Set = []
        
        self.act_loss = 0
        self.Q_mean = []
        
        self.loss_act_set =  []
        self.Q_set = []
        
        self.one_nei_set = []
        self.two_nei_set = []

        self.direction_angle = 0 
        self.dis_move = 20        
        
        
        """
        UAV本地DDPG模型：
        
        """
        #DDPG结构
        self.Actor_Net_E = Actor(State_Dim, 48, 72, 32, Beam_Num) #Actor evaluation网络
        self.Actor_Net_T = Actor(State_Dim, 48, 72, 32, Beam_Num) #Actor taregt网络
        
        self.Critic_Net_E = Critic((State_Dim * 2), 64, 100, 64, 1) #Critic evaluation网络
        self.Critic_Net_T = Critic((State_Dim * 2), 64, 100, 64, 1) #Critic taregt网络
        
        self.Ex_Pool = ReplayMemory(Pool_Capacity) #经验回放池
        
        #两个DQN的优化器
        self.actor_optim = optim.Adam(self.Actor_Net_E.parameters(), lr = LR_Actor)
        self.critic_optim = optim.Adam(self.Critic_Net_E.parameters(), lr = LR_Critic)

        #本地权重和控制变量参数        
        self.Net_Weight_A = [0 for m in range(NNet_Layer_Num)]  #神经网络权重参数actor
        self.Net_Weight_Grad_A = [0 for n in range(NNet_Layer_Num)]  #神经网络权重梯度actor
        self.Net_Weight_Grad_A_1 = [0 for n in range(NNet_Layer_Num)]  #神经网络权重梯度actor
        self.Net_Weight_Grad_A_2 = [0 for n in range(NNet_Layer_Num)]  #神经网络权重梯度actor
        self.Net_Weight_Grad_A_Set = []  #神经网络权重梯度记录
        
        self.Net_Weight_C = [0 for m in range(NNet_Layer_Num)]  #神经网络权重参数critic
        self.Net_Weight_Grad_C = [0 for n in range(NNet_Layer_Num)]  #神经网络权重梯度critic
        self.Net_Weight_Grad_C_1 = [0 for n in range(NNet_Layer_Num)]  #神经网络权重梯度critic
        self.Net_Weight_Grad_C_2 = [0 for n in range(NNet_Layer_Num)]  #神经网络权重梯度critic
        self.Net_Weight_Grad_C_Set = []  #神经网络权重梯度记录
        
        self.Control_Value_A = [0 for n in range(NNet_Layer_Num)] #本地控制变量actor
        self.Control_Value_C = [0 for n in range(NNet_Layer_Num)] #本地控制变量critic
        
        self.Control_Value_A_copy = [0 for n in range(NNet_Layer_Num)] #本地控制变量actor备份
        self.Control_Value_C_copy = [0 for n in range(NNet_Layer_Num)] #本地控制变量critic备份
        
        #保存当前全局参数
        self.Global_Weight_A = [0 for n in range(NNet_Layer_Num)]
        self.Global_Weight_C = [0 for n in range(NNet_Layer_Num)]
        self.Global_Control_Value_A = [0 for n in range(NNet_Layer_Num)]
        self.Global_Control_Value_C = [0 for n in range(NNet_Layer_Num)]
        
        #待发送本地参数
        self.Delta_Control_Value_A = [0 for n in range(NNet_Layer_Num)]
        self.Delta_Control_Value_C = [0 for n in range(NNet_Layer_Num)]
        
        self.Delta_Net_Weight_A = [0 for n in range(NNet_Layer_Num)]
        self.Delta_Net_Weight_C = [0 for n in range(NNet_Layer_Num)]
        
      
    
    def move(self):   #用户移动函数        
        self.location[0] = self.location[0] + self.dis_move * math.cos(self.direction_angle)
        self.location[1] = self.location[1] + self.dis_move * math.sin(self.direction_angle)
        
        self.direction_angle = self.direction_angle + random.uniform(-((math.pi) / 4) , ((math.pi) / 4))
        if self.direction_angle < 0:
            self.direction_angle = self.direction_angle + 2 * (math.pi)
            
        """越界返回"""
        if self.location[0] > Length:
            self.location[0] -= 30
        elif self.location[0] < 0:
            self.location[0] += 30 
            
        if self.location[1] > Length:
            self.location[1] -= 30
        elif self.location[1] < 0:
            self.location[1] += 30
    
    #每个时隙将碰撞次数初始化
    def Col_Num_Init(self): 
        self.state_pre = [(State_weight_1 * i + State_weight_2 * j + State_weight_3* k) for i, j, k in zip(self.beam_nei_t, self.col_num, self.beam_nei_o)]
        for i in range(len(self.state_pre)):
            if self.state_pre[i] == 0:
                self.state_pre[i] += 2 
        self.col_num = [0 for h in range(Sec_Num)] #初始化碰撞次数
    
    #初始化一跳邻居表
    def Clear_Nei(self):
        self.nei_beam_one = [0 for n in range(Sec_Num)]
        
    def Act_Trans(self, act_ten):
        Act_Sample = act_ten.tolist()
        Act_Tran = []
        for act_sam in Act_Sample:
            act_tem = [2 for i in range(State_Dim)]
            for j in act_sam:
                act_tem[int(j) - 1] = 20
            Act_Tran.append(act_tem)
        Act_Tran = torch.tensor(Act_Tran)
        return Act_Tran
        
    
    #每个通信轮次的开始根据一跳邻居表初始化动作
    def act_init(self):
        act_ini = []
        nei_beam_list = []
        nei_beam_index = []
        fla = 1
        for n in self.nei_beam_one:
            nei_beam_list.append(n)
            nei_beam_index.append(fla)
            fla += 1

        for m in range(Beam_Num):
            n_index = nei_beam_list.index(max(nei_beam_list))
            act_ini.append(nei_beam_index[n_index])
            nei_beam_list.pop(n_index)
            nei_beam_index.pop(n_index)
        act_ini = tuple(act_ini)
        self.Clear_Nei()
        return act_ini
    
                
    #每个通信周期后，初始化邻居表(由于节点移动)，
    def Par_Init_Class(self):
        for nei_one_i in self.nei_one:
            self.nei_beam_one[nei_one_i[1] - 1] += 1   #为每个周期的动作初始化做准备
            
        self.nei_one = []   #一跳邻居表
        self.nei_two = []   #二跳邻居表
             
        self.beam_nei_o = [0 for k in range(Sec_Num)]  #一跳邻居分布扇区
        self.beam_nei_t = [0 for j in range(Sec_Num)]  #二跳邻居分布扇区
        
        self.beam_nei = [0 for g in range(Sec_Num)] #邻居分布
        self.col_num = [0 for h in range(Sec_Num)] #碰撞次数
        
        self.idle_num = 0  #每个时隙错过的邻居
        self.slot_num = 0
        
        self.state_pre = [2 for p in range(State_Dim)]
        self.state_nex = [2 for m in range(State_Dim)]
        self.reward = 1 #计算每次的奖励值
        self.action = [0 for w in range(Beam_Num)]
        self.Loss_Set = []
        
        
        self.Global_Weight_A = [0 for n in range(NNet_Layer_Num)]
        self.Global_Weight_C = [0 for n in range(NNet_Layer_Num)]
        self.Global_Control_Value_A = [0 for n in range(NNet_Layer_Num)]
        self.Global_Control_Value_C = [0 for n in range(NNet_Layer_Num)]
        
        
        self.Delta_Control_Value_A = [0 for n in range(NNet_Layer_Num)]
        self.Delta_Control_Value_C = [0 for n in range(NNet_Layer_Num)]
        
        self.Delta_Net_Weight_A = [0 for n in range(NNet_Layer_Num)]
        self.Delta_Net_Weight_C = [0 for n in range(NNet_Layer_Num)]
        
        
        
    #更新邻居扇区分布，计算奖励以及下一个状态
    def Cal_Beam_Nei(self):
        nei_o_num = sum(self.beam_nei_o)
        
        self.beam_nei_o = [0 for k in range(Sec_Num)]  
        self.beam_nei_t = [0 for j in range(Sec_Num)]
        
        for nei_one_i in self.nei_one:
            self.beam_nei_o[nei_one_i[1] - 1] += 1
        for nei_two_i in self.nei_two:
            self.beam_nei_t[nei_two_i[1] - 1] += 1
        
        self.reward = Beta_1 * (sum(self.beam_nei_o) - nei_o_num) + Beta_2 * (sum(self.col_num)) + Beta_3 * sum(self.beam_nei_t)

        self.state_nex = [(State_weight_1 * i + State_weight_2 * j + State_weight_3 * k) for i, j, k in zip(self.beam_nei_t, self.col_num, self.beam_nei_o)]
        
        for i in range(len(self.state_nex)):
            if self.state_nex[i] == 0:
                self.state_nex[i] += 2 
                
    #保存一组数据
    def Save_Data(self):
        self.Ex_Pool.push(self.state_pre, self.action, self.reward, self.state_nex)
            
    #网络参数初始化
    def Par_Init(self, Par_W_A, Par_W_C, Par_C_A, Par_C_C):
        for w_a in range(len(Par_W_A)):
            self.Global_Weight_A[w_a] = Par_W_A[w_a]
        for w_c in range(len(Par_W_C)):
            self.Global_Weight_C[w_c] = Par_W_C[w_c]
        for c_a in range(len(Par_C_A)):
            self.Global_Control_Value_A[c_a] = Par_C_A[c_a]
        for c_c in range(len(Par_C_C)):
            self.Global_Control_Value_C[c_c] = Par_C_C[c_c]
            
        Flag_g = 0

        for name, parameters in self.Critic_Net_T.state_dict().items():
            if "weight" in name:
                self.Critic_Net_T.state_dict()[name].copy_(Par_W_C[Flag_g])
                Flag_g = Flag_g + 1 
        Flag_g = 0

        for name, parameters in self.Actor_Net_T.state_dict().items():
            if "weight" in name:
                self.Actor_Net_T.state_dict()[name].copy_(Par_W_A[Flag_g])
                Flag_g = Flag_g + 1
                
    def Extract_Par_C(self): #参数提取输出函数，用于参数的聚合
        Flag_p = 0
        for name,param in self.Critic_Net_E.named_parameters():    
            if "weight" in name:
                self.Net_Weight_C[Flag_p] = param.float()              #采集critic权重
                Flag_p = Flag_p + 1
     
    def Extract_Par_A(self): #参数提取输出函数，用于参数的聚合
        Flag_p = 0
        for name,param in self.Actor_Net_E.named_parameters():    
            if "weight" in name:
               # print("***",param)
                self.Net_Weight_A[Flag_p] = param.float()              #采集actor权重
                Flag_p = Flag_p + 1
    
    def Extract_Par_G_C(self, m): #参数提取输出函数，用于参数的聚合
        if m == 0:
            Flag_p = 0
            for name,param in self.Critic_Net_E.named_parameters():    
                if "weight" in name:
                    self.Net_Weight_Grad_C[Flag_p] = param.grad.float()    #采集critic梯度
                    Flag_p = Flag_p + 1
            self.Net_Weight_Grad_C_Set.append(self.Net_Weight_Grad_C)
        elif m == 1:
            Flag_p = 0
            for name,param in self.Critic_Net_E.named_parameters():    
                if "weight" in name:
                    self.Net_Weight_Grad_C_1[Flag_p] = param.grad.float()    #采集critic梯度
                    Flag_p = Flag_p + 1
            self.Net_Weight_Grad_C_Set.append(self.Net_Weight_Grad_C_1)
        elif m == 2:
            Flag_p = 0
            for name,param in self.Critic_Net_E.named_parameters():    
                if "weight" in name:
                    self.Net_Weight_Grad_C_2[Flag_p] = param.grad.float()    #采集critic梯度
                    Flag_p = Flag_p + 1
            self.Net_Weight_Grad_C_Set.append(self.Net_Weight_Grad_C_2)
                
    def Extract_Par_G_A(self, n): #参数提取输出函数，用于参数的聚合 
        if n == 0:
            Flag_p = 0
            for name_1,param_1 in self.Actor_Net_E.named_parameters():    
                if "weight" in name_1:
                    self.Net_Weight_Grad_A[Flag_p] = param_1.grad.float()    #采集actor梯度
                    Flag_p = Flag_p + 1
            self.Net_Weight_Grad_A_Set.append(self.Net_Weight_Grad_A)
        elif n == 1:
            Flag_p = 0
            for name_1,param_1 in self.Actor_Net_E.named_parameters():    
                if "weight" in name_1:
                    self.Net_Weight_Grad_A_1[Flag_p] = param_1.grad.float()    #采集actor梯度
                    Flag_p = Flag_p + 1
            self.Net_Weight_Grad_A_Set.append(self.Net_Weight_Grad_A_1)
        elif n == 2:
            Flag_p = 0
            for name_1,param_1 in self.Actor_Net_E.named_parameters():    
                if "weight" in name_1:
                    self.Net_Weight_Grad_A_2[Flag_p] = param_1.grad.float()    #采集actor梯度
                    Flag_p = Flag_p + 1
            self.Net_Weight_Grad_A_Set.append(self.Net_Weight_Grad_A_2)
    
    #本地控制变量初始化，即全部置为零
    def Control_Value_Init(self): 
        self.Extract_Par_A()
        self.Extract_Par_C()
        self.Global_Weight_A = [torch.zeros_like(p) for p in self.Net_Weight_A if p.requires_grad]
        self.Global_Weight_C = [torch.zeros_like(q) for q in self.Net_Weight_C if q.requires_grad]
        
        self.Control_Value_A = [torch.zeros_like(p) for p in self.Net_Weight_A if p.requires_grad]
        self.Control_Value_C = [torch.zeros_like(q) for q in self.Net_Weight_C if q.requires_grad]
        
        self.Global_Control_Value_A = [torch.zeros_like(p) for p in self.Net_Weight_A if p.requires_grad]
        self.Global_Control_Value_C = [torch.zeros_like(q) for q in self.Net_Weight_C if q.requires_grad]

    #K轮训练后更新本地控制变量
    def Control_Value_Update(self, K): 
        for flag_i in range(len(self.Net_Weight_A)):
            self.Control_Value_A_copy[flag_i] = self.Control_Value_A[flag_i]       
            self.Control_Value_A[flag_i] = self.Control_Value_A[flag_i] - self.Global_Control_Value_A[flag_i] +  (self.Global_Weight_A[flag_i] - self.Net_Weight_A[flag_i]) / K        
        for flag_j in range(len(self.Net_Weight_C)):
            self.Control_Value_C_copy[flag_j] = self.Control_Value_C[flag_j]
            self.Control_Value_C[flag_j] = self.Control_Value_C[flag_j] - self.Global_Control_Value_C[flag_j] +  (self.Global_Weight_C[flag_j] - self.Net_Weight_C[flag_j]) / K
    
    #传输参数准备
    def Par_Com_Pre(self):
        for flag_k in range(len(self.Control_Value_A)):
            self.Delta_Control_Value_A[flag_k] = (self.Control_Value_A[flag_k] - self.Control_Value_A_copy[flag_k])
        for flag_l in range(len(self.Control_Value_C)):
            self.Delta_Control_Value_C[flag_l] = (self.Control_Value_C[flag_l] - self.Control_Value_C_copy[flag_l])
    
    #更新一步
    def Extra_Step_C(self):
        Flag_g = 0
        for name, parameters in self.Critic_Net_E.state_dict().items():
            if "weight" in name:
                self.Critic_Net_E.state_dict()[name].copy_(parameters - LR_Critic * (self.Global_Control_Value_C[Flag_g] - self.Control_Value_C[Flag_g]))
                Flag_g = Flag_g + 1 
    #更新一步            
    def Extra_Step_A(self):
        Flag_g = 0
        for name, parameters in self.Actor_Net_E.state_dict().items():
            if "weight" in name:
                self.Actor_Net_E.state_dict()[name].copy_(parameters - LR_Actor * (self.Global_Control_Value_A[Flag_g] - self.Control_Value_A[Flag_g]))
                Flag_g = Flag_g + 1
    
    def Act_tra(self, act_one):
        act_one = act_one.tolist()
        act_tra = [2 for i in range(State_Dim)]
        for j in act_one:
            act_tra[int(j) - 1] = 20
        act_tra = torch.tensor(act_tra)
        return act_tra
        
        
    #使用K近邻思想，将连续动作空间映射到离散动作空间，挑选出最好的动作
    def act(self, s_t_set, cho, tens, com_num, flag):  #s_t为当前状态，cho为选择1:evaluate网络还是 2:target网络计算
        if tens == 0:
            s_t_set = torch.tensor(s_t_set, dtype=torch.float).unsqueeze(0) #unsqueeze升维函数
        a_t_set = []
        act_loss = 0
        
        for s_t in s_t_set: 
            if cho == 1:
                a_t = self.Actor_Net_E(s_t).squeeze(0).detach().numpy()               

            elif cho == 2:
                a_t = self.Actor_Net_T(s_t).squeeze(0).detach().numpy() #squeeze降维函数
                
            a_t = a_t.tolist()
                        
            #计算出K个最邻近可行解
            a_dis = [0 for l in range(len(Act_Space))]
            a_set = [] 
            a_space = list(combinations([(v+1) for v in range(Sec_Num)], Beam_Num))
            for a_i in range(len(a_space)):
                a_dis_i = 0
                for i in range(Beam_Num):
                    a_dis_i = a_dis_i + ((a_t[i] - a_space[a_i][i]) ** 2)
                a_dis_i = math.sqrt(a_dis_i)
                a_dis[a_i] = a_dis_i
            
            for p in range(K):
                a_index = a_dis.index(min(a_dis))
                a_set.append(a_space[a_index])
                a_space.pop(a_index)
                a_dis.pop(a_index)
                
            #计算最大Q值的动作
            C_Net_Q = []
            for a_j in a_set:
                a_j = torch.tensor(a_j, dtype=torch.float)
                C_Q_Value = self.Critic_Net_E(s_t, self.Act_tra(a_j), 0)
                C_Q_Value = float(C_Q_Value.detach().numpy())
                C_Net_Q.append(C_Q_Value)
            Max_Q_Index = C_Net_Q.index(max(C_Net_Q))
            a_t_set.append(a_set[Max_Q_Index])
            
            a_dis = 0
            for i in range(Beam_Num):
                a_dis = a_dis + ((a_t[i] - a_set[Max_Q_Index][i]) ** 2)
            act_loss = act_loss + math.sqrt(a_dis)
        
        self.act_loss = act_loss
            
        if len(a_t_set) == 1:
            if (random.random() < max([0.3, (1 - com_num * 0.05)])) and (flag == 0):
                return random.choice(Act_Space)#随机探索
            else:
                return a_t_set[0]
        else:
            return a_t_set

    
    #采样数据
    def Sample_data(self):      
        samples = self.Ex_Pool.sample(Batch_Size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float).view(Batch_Size,-1)
        a0 = torch.tensor(a0, dtype=torch.float).view(Batch_Size,-1)
        r1 = torch.tensor(r1, dtype=torch.float).view(Batch_Size,-1)
        s1 = torch.tensor(s1, dtype=torch.float).view(Batch_Size,-1) 
        
        return s0,a0,r1,s1
    
    #训练网络
    def learn(self):
        
        if len(self.Ex_Pool.memory) < Batch_Size:
            return 0
        
        #采样数据
        s_0, a_0, r_0, s_0_nex = self.Sample_data()
        s_1, a_1, r_1, s_1_nex = self.Sample_data()
        s_2, a_2, r_2, s_2_nex = self.Sample_data()
        s_3, a_3, r_3, s_3_nex = self.Sample_data()
        
        def Critic_learn():  
            self.Extract_Par_C() #提取权重参数
            
            a_0_nex = self.act(s_0_nex, 2, 1, 1, 1)
            a_0_nex_t = torch.tensor(a_0_nex, dtype=torch.float)
            
            
            y_true = r_0 + GAMMA * self.Critic_Net_T(s_0_nex, self.Act_Trans(a_0_nex_t), 1).detach()
            y_pred = self.Critic_Net_E(s_0, self.Act_Trans(a_0), 1)
                        
            loss = F.smooth_l1_loss(y_pred, y_true)   
            
            self.critic_optim.zero_grad()
            loss.backward()
            
            #nn.utils.clip_grad_norm_(parameters = self.Critic_Net_E.parameters(), max_norm=20, norm_type=2)
            
            self.critic_optim.step()
            
            #MAML第二步            
            a_1_nex = self.act(s_1_nex, 2, 1, 1, 1)
            a_1_nex_t = torch.tensor(a_1_nex, dtype=torch.float)
            
            y_true_1 = r_1 + GAMMA * self.Critic_Net_T(s_1_nex, self.Act_Trans(a_1_nex_t), 1).detach()
            y_pred_1 = self.Critic_Net_E(s_1, self.Act_Trans(a_1), 1)
            
            loss_1 = F.smooth_l1_loss(y_pred_1, y_true_1)
                       
            self.critic_optim.zero_grad()
            loss_1.backward()
            
            #nn.utils.clip_grad_norm_(parameters = self.Critic_Net_E.parameters(), max_norm=20, norm_type=2)
            
            self.critic_optim.step()
            
            self.Extract_Par_G_C(0)
            
            Flag_1 = 0
            for name, parameters in self.Critic_Net_E.state_dict().items():
                if "weight" in name:
                    Theta_New = self.Net_Weight_C[Flag_1] + Sigma * self.Net_Weight_Grad_C[Flag_1]
                    self.Critic_Net_E.state_dict()[name].copy_(Theta_New)
                    Flag_1 = Flag_1 + 1 

            a_2_nex = self.act(s_2_nex, 2, 1, 1, 1)
            a_2_nex_t = torch.tensor(a_2_nex, dtype=torch.float)
            
            y_true_2 = r_2 + GAMMA * self.Critic_Net_T(s_2_nex, self.Act_Trans(a_2_nex_t), 1).detach()
            y_pred_2 = self.Critic_Net_E(s_2, self.Act_Trans(a_2), 1)
            
            loss_2 = F.smooth_l1_loss(y_pred_2, y_true_2)
                        
            self.critic_optim.zero_grad()
            loss_2.backward()
            
            #nn.utils.clip_grad_norm_(parameters = self.Critic_Net_E.parameters(), max_norm=20, norm_type=2)
            
            self.critic_optim.step()
            self.Extract_Par_G_C(1)            

            Flag_2 = 0
            for name, parameters in self.Critic_Net_E.state_dict().items():
                if "weight" in name:
                    Theta_New = self.Net_Weight_C[Flag_2] - Sigma * self.Net_Weight_Grad_C[Flag_2]
                    self.Critic_Net_E.state_dict()[name].copy_(Theta_New)
                    Flag_2 = Flag_2 + 1 
            
            y_true_3 = r_2 + GAMMA * self.Critic_Net_T(s_2_nex, self.Act_Trans(a_2_nex_t), 1).detach()
            y_pred_3 = self.Critic_Net_E(s_2, self.Act_Trans(a_2), 1)
            
            loss_3 = F.smooth_l1_loss(y_pred_3, y_true_3)
                        
            self.critic_optim.zero_grad()
            loss_3.backward()
            
            #nn.utils.clip_grad_norm_(parameters = self.Critic_Net_E.parameters(), max_norm=20, norm_type=2)
            
            self.critic_optim.step()
            self.Extract_Par_G_C(2)   
            
            Flag_3 = 0
            for name, parameters in self.Critic_Net_E.state_dict().items():
                if "weight" in name:
                    self.Net_Weight_Grad_C[Flag_3] = self.Net_Weight_Grad_C[Flag_3] - LR_MAML * ((self.Net_Weight_Grad_C_1[Flag_3] - self.Net_Weight_Grad_C_2[Flag_3]) / (2 * Sigma))
                    Flag_3 = Flag_3 + 1 
            
            Flag = 0
            for name, parameters in self.Critic_Net_E.state_dict().items():
                if "weight" in name:
                    Theta_New = self.Net_Weight_C[Flag] - LR_MAML * self.Net_Weight_Grad_C[Flag]
                    self.Critic_Net_E.state_dict()[name].copy_(Theta_New)
                    Flag = Flag + 1 
                    
            self.Extra_Step_C()
                      
            a_3_nex = self.act(s_3_nex, 2, 1, 1, 1)
            a_3_nex_t = torch.tensor(a_3_nex, dtype=torch.float)
            
            y_true_4 = r_3 + GAMMA * self.Critic_Net_T(s_3_nex, self.Act_Trans(a_3_nex_t), 1).detach()
            
            y_pred_4 = self.Critic_Net_E(s_3, self.Act_Trans(a_3), 1)
            
            self.Q_set.append([s_3.tolist(),self.Act_Trans(a_3).tolist(),y_pred_4.tolist()])
            
            loss_4 = F.smooth_l1_loss(y_pred_4, y_true_4)
            
            self.Loss_Set.append(loss_4.tolist())                     
                                                                                                                                                                        
        def Actor_learn():
            self.Extract_Par_A()
            a_1_e = torch.tensor(self.act(s_0, 1, 1, 1, 1), dtype=torch.float)
            Q_mean = torch.mean(-1 * (self.Critic_Net_E(s_0, self.Act_Trans(a_1_e), 1)))
            self.Q_mean.append(float(Q_mean))
            loss = torch.tensor(Q_mean + torch.tensor(self.act_loss))
            self.loss_act_set.append(float(loss))
            loss.requires_grad_(True)
            
            self.actor_optim.zero_grad()
            loss.backward()  
            #nn.utils.clip_grad_norm_(parameters = self.Actor_Net_E.parameters(), max_norm=20, norm_type=2)
            self.actor_optim.step()
            self.Extra_Step_A()
            
        
        def Soft_update(net_target, net):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - Tau) + param.data * Tau) 
        
        Critic_learn()
        Actor_learn()
        Soft_update(self.Critic_Net_T, self.Critic_Net_E)
        Soft_update(self.Actor_Net_T, self.Actor_Net_E)
               

            
def PFL():
    Comm_Num = 50 #通信周期数量
    Slot_Num  = 120 #每个通信周期邻居发现阶段的时隙数量
    
    UAV_NNet_Weight_A_List = [0 for p in range(UAV_Num)]        #所有节点的actor神经网络权重参数集合
    UAV_NNet_Weight_C_List = [0 for p in range(UAV_Num)]        #所有节点的critic神经网络权重参数集合
    UAV_Control_Value_A_List = [0 for p in range(UAV_Num)]      #所有节点的actor控制变量参数集合
    UAV_Control_Value_C_List = [0 for p in range(UAV_Num)]      #所有节点的critic控制变量参数集合
    
    #全局参数
    UAV_Par_A_List = [0 for n in range(NNet_Layer_Num)]
    UAV_Par_C_List = [0 for n in range(NNet_Layer_Num)]
    UAV_Con_A_List = [0 for n in range(NNet_Layer_Num)]
    UAV_Con_C_List = [0 for n in range(NNet_Layer_Num)]
    
    '''所有可能的链路'''
    Node_list = [(i+1) for i in range(UAV_Num)]  
    Link_Space = list(combinations(Node_list, 2))
    
    
    UAV_List = [] #无人机群集合
    
    for e in range(UAV_Num): #创建无人机群
        UAV_List.append(UAV(e+1)) #w无人机群编号1,...
        UAV_List[e].Control_Value_Init()
    
    
    def Par_Poly(P_A, P_C, C_A, C_C): #参数聚合
        for i in range(len(P_A[0])):
            Par_Temp = 0 
            for j in range(UAV_Num):
                Par_Temp = P_A[j][i] + Par_Temp
            Par_Temp = Par_Temp / UAV_Num
            UAV_Par_A_List[i] = Par_Temp
        for i in range(len(P_C[0])):
            Par_Temp = 0 
            for j in range(UAV_Num):
                Par_Temp = P_C[j][i] + Par_Temp
            Par_Temp = Par_Temp / UAV_Num 
            UAV_Par_C_List[i] = Par_Temp
        for i in range(len(C_A[0])):
            Par_Temp = 0 
            for j in range(UAV_Num):
                Par_Temp = C_A[j][i] + Par_Temp
            Par_Temp = Par_Temp / UAV_Num
            UAV_Con_A_List[i] = Par_Temp
        for i in range(len(C_C[0])):
            Par_Temp = 0 
            for j in range(UAV_Num):
                Par_Temp = C_C[j][i] + Par_Temp
            Par_Temp = Par_Temp / UAV_Num
            UAV_Con_C_List[i] = Par_Temp
    
  
    link_per_set = []   
            
    #学习主程序
    for com_r in range(Comm_Num):
        print("Comm_Round",com_r + 1)
        Link_Set = [] #所有可建立的链路
        Link_Dis = [] #链路长度
        Link_Sec = [] #链路两端节点对应扇区
        Link_Set_Exi = [] #已建立连接的链路
        Link_percent = [0 for i in range(Slot_Num)] #已建立的链路比例 
        
        Link_matrix = [[0 for i in range(UAV_Num)] for j in range(UAV_Num)] #邻接矩阵
        
        for uav_i in range(UAV_Num):
            UAV_List[uav_i].Par_Init_Class()
            UAV_List[uav_i].move()
                
        if com_r > 0:
            for uav_k in UAV_List:
                uav_k.Par_Init(UAV_Par_A_List, UAV_Par_C_List, UAV_Con_A_List, UAV_Con_C_List) #本地模型参数初始化
        
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
    
        for link_k in range(len(Link_Set)):
            node_a = Link_Set[link_k][0] - 1
            node_b = Link_Set[link_k][1] - 1
    
            UAV_List[node_a].beam_nei[Link_Sec[link_k][0] - 1] += 1
            UAV_List[node_b].beam_nei[Link_Sec[link_k][1] - 1] += 1
            
        for slot_i in range(Slot_Num):                      
            Act_all_node = []     
            if (com_r > 0) and (slot_i == 0):
                for uav_m in UAV_List: 
                    uav_m.idle_num = 0 
                    act_uav_j = uav_m.act_init()
                    uav_m.action_set.append(act_uav_j)
                    Act_all_node.append(act_uav_j)
            else:            
                for uav_j in range(UAV_Num):
                    UAV_List[uav_j].idle_num = 0 
                    UAV_List[uav_j].slot_num += 1
                    UAV_List[uav_j].Col_Num_Init()
                    act_uav_i = UAV_List[uav_j].act(UAV_List[uav_j].state_pre, 1, 0, com_r, 0)
                    UAV_List[uav_j].action_set.append(act_uav_i)
                    UAV_List[uav_j].action = act_uav_i                          
                    Act_all_node.append(act_uav_i)
            
                            
            #计算一跳邻居
            for link_j in range(len(Link_Set)):
                
                node_c = Link_Set[link_j][0] - 1
                node_d = Link_Set[link_j][1] - 1
                
                if (Link_Sec[link_j][0] in Act_all_node[node_c]) and (Link_Sec[link_j][1] in Act_all_node[node_d]):
                    nei_num_1 = UAV_List[node_c].beam_nei[Link_Sec[link_j][0] - 1] 
                    nei_num_2 = UAV_List[node_d].beam_nei[Link_Sec[link_j][1] - 1]
                    nei_num_sum = (nei_num_1 + nei_num_2)
                    
                    if nei_num_sum > 10:
                        nei_num_sum = 10
                    elif nei_num_sum < 2:
                        nei_num_sum = 2
                        
                    col_pro =  (2 ** nei_num_sum - 4) / (2 ** 10 + 200)#碰撞概率
                    
                    #满足条件则修改邻居表
                    if random.random() >= col_pro:    
                        if Link_Set[link_j] not in Link_Set_Exi:
                            Link_Set_Exi.append(Link_Set[link_j])
                        
                        #修改一跳邻居表
                        if [Link_Set[link_j][1],Link_Sec[link_j][0]] not in UAV_List[node_c].nei_one:
                            if [Link_Set[link_j][1],Link_Sec[link_j][0]] in UAV_List[node_c].nei_two:
                                UAV_List[node_c].nei_two.remove([Link_Set[link_j][1],Link_Sec[link_j][0]])    
                            UAV_List[node_c].nei_one.append([Link_Set[link_j][1],Link_Sec[link_j][0]])
                            Link_matrix[node_c][node_d] = 1
                        
                        if [node_c + 1,Link_Sec[link_j][1]] not in UAV_List[node_d].nei_one:
                            if [node_c + 1,Link_Sec[link_j][1]] in UAV_List[node_d].nei_two:
                                UAV_List[node_d].nei_two.remove([node_c + 1,Link_Sec[link_j][1]])    
                            UAV_List[node_d].nei_one.append([node_c + 1,Link_Sec[link_j][1]])
                            Link_matrix[node_d][node_c] = 1
    
                        for nei_one_i in UAV_List[node_c].nei_one:
                            if nei_one_i[0] == node_d + 1:
                                continue
                            else:
                                dis_j, sec_j = Link_Com(UAV_List[node_d], UAV_List[nei_one_i[0] - 1])
                                if ((dis_j < Dis_Thr) and ([nei_one_i[0], sec_j] not in UAV_List[node_d].nei_one)) and ([nei_one_i[0], sec_j] not in UAV_List[node_d].nei_two):
                                    UAV_List[node_d].nei_two.append([nei_one_i[0], sec_j])
                                
                        for nei_one_j in UAV_List[node_d].nei_one:
                            if nei_one_j[0] == node_c + 1:
                                continue
                            else:
                                dis_k, sec_k = Link_Com(UAV_List[node_c], UAV_List[nei_one_j[0] - 1])
                                if ((dis_k < Dis_Thr) and ([nei_one_j[0], sec_k] not in UAV_List[node_c].nei_one)) and ([nei_one_j[0], sec_k] not in UAV_List[node_c].nei_two):
                                    UAV_List[node_c].nei_two.append([nei_one_j[0], sec_k])
                    else:
                        UAV_List[node_c].col_num[Link_Sec[link_j][0] - 1] += 1
                        UAV_List[node_d].col_num[Link_Sec[link_j][1] - 1] += 1
                                    
            #更新节点扇区-邻居分布数据,计算状态、奖励等数据并保存        
            for uav_i in range(UAV_Num):
                UAV_List[uav_i].Cal_Beam_Nei()
                UAV_List[uav_i].Save_Data()
                
            if CG.CG(Link_matrix, UAV_Num) == 1:
                print("连通图形成",slot_i)
                
            #学习，神经网络训练  
            for uav_p in UAV_List:
                uav_p.learn()
            link_percent_i = (len(Link_Set_Exi)/len(Link_Set)) * 100
            Link_percent[slot_i] = link_percent_i
            link_per_set.append(link_percent_i)
            print("Slot_Num", slot_i + 1, link_percent_i) 
                
                                      
        #联邦学习，参数聚合、参数下发
        flag_j = 0        
        for uav_n in UAV_List:
            uav_n.Extract_Par_A()
            uav_n.Extract_Par_C()#参数提取
    
            uav_n.Control_Value_Update(Slot_Num)
            uav_n.Par_Com_Pre()
            
            UAV_NNet_Weight_A_List[flag_j] = uav_n.Net_Weight_A        #所有节点的actor神经网络权重参数集合
            UAV_NNet_Weight_C_List[flag_j] = uav_n.Net_Weight_C       #所有节点的critic神经网络权重参数集合
            UAV_Control_Value_A_List[flag_j] = uav_n.Delta_Control_Value_A       #所有节点的actor控制变量参数集合
            UAV_Control_Value_C_List[flag_j] = uav_n.Delta_Control_Value_C
            
            flag_j = flag_j + 1
        
        Par_Poly(UAV_NNet_Weight_A_List, UAV_NNet_Weight_C_List, UAV_Control_Value_A_List, UAV_Control_Value_C_List) #参数聚合
        
        
    
    
        
       
            
            
                    
                    
                
                
                
    
          
    
    
    
    
