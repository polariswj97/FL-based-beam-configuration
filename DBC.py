"""
Created on Thu Jun  1 16:07:29 2023
FL + DDPG
@author: Jian Wang
"""
import CG
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
Dis_Thr = 400    

LR_Actor = 0.001    
LR_Critic = 0.001   
LR_Global = 0.001   
Tau = 0.05          
GAMMA = 0.9          
Epsilon = 0.9       

Beta_1 = 500         
Beta_2 = 60        
Beta_3 = 60         

State_weight_1 = 60
State_weight_2 = 60
State_weight_3 = 10

Pool_Capacity = 1000 
Batch_Size = 32     
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
    PL = 32.4 + 17.3 * np.log10(F_UAV) + 20 * np.log10(dis)
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
        self.capacity = capacity 
        self.memory = [] 
        
    def push(self, *args):
        if len(self.memory) == self.capacity:
            self.memory.pop(0) 
        self.memory.append(args)
           
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def data_num(self):
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
        self.location = np.random.randint(0, Length, 2)

        self.nei_one = []   
        self.nei_two = []
        
        self.beam_nei_o = [0 for k in range(Sec_Num)]  
        self.beam_nei_t = [0 for j in range(Sec_Num)] 
        
        self.nei_beam_one = [0 for n in range(Sec_Num)] 

        self.beam_nei = [0 for g in range(Sec_Num)] 
        self.col_num = [0 for h in range(Sec_Num)] 
        
        self.slot_num = 0 
        
        self.reward = 0 
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

        self.Actor_Net_E = Actor(State_Dim, 48, 72, 32, Beam_Num) 
        self.Actor_Net_T = Actor(State_Dim, 48, 72, 32, Beam_Num) 
        
        self.Critic_Net_E = Critic((State_Dim * 2), 64, 100, 64, 1) 
        self.Critic_Net_T = Critic((State_Dim * 2), 64, 100, 64, 1) 
        
        self.Ex_Pool = ReplayMemory(Pool_Capacity) 

        self.actor_optim = optim.Adam(self.Actor_Net_E.parameters(), lr = LR_Actor)
        self.critic_optim = optim.Adam(self.Critic_Net_E.parameters(), lr = LR_Critic)
          
    def move(self):     
        self.location[0] = self.location[0] + self.dis_move * math.cos(self.direction_angle)
        self.location[1] = self.location[1] + self.dis_move * math.sin(self.direction_angle)
        
        self.direction_angle = self.direction_angle + random.uniform(-((math.pi) / 4) , ((math.pi) / 4))
        if self.direction_angle < 0:
            self.direction_angle = self.direction_angle + 2 * (math.pi)

        if self.location[0] > Range:
            self.location[0] -= 30
        elif self.location[0] < 0:
            self.location[0] += 30 
            
        if self.location[1] > Range:
            self.location[1] -= 30
        elif self.location[1] < 0:
            self.location[1] += 30
            
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
    
    def Col_Num_Init(self): 
        self.state_pre = [(State_weight_1 * i + State_weight_2 * j + State_weight_3* k) for i, j, k in zip(self.beam_nei_t, self.col_num, self.beam_nei_o)]
        for i in range(len(self.state_pre)):
            if self.state_pre[i] == 0:
                self.state_pre[i] += 2 
        self.col_num = [0 for h in range(Sec_Num)]

    def Clear_Nei(self):
        self.nei_beam_one = [0 for n in range(Sec_Num)]  
        
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
    
    def Par_Init_Class(self):
        for nei_one_i in self.nei_one:
            self.nei_beam_one[nei_one_i[1] - 1] += 1 
            
        self.nei_one = []   
        self.nei_two = []   
             
        self.beam_nei_o = [0 for k in range(Sec_Num)]  
        self.beam_nei_t = [0 for j in range(Sec_Num)]  
        
        self.beam_nei = [0 for g in range(Sec_Num)] 
        self.col_num = [0 for h in range(Sec_Num)] 
         
        self.slot_num = 0
        
        self.state_pre = [2 for p in range(State_Dim)]
        self.state_nex = [2 for m in range(State_Dim)]
        self.reward = 1 
        self.action = [2 for w in range(Beam_Num)]
        self.Loss_Set = []

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

    def Save_Data(self):
        self.Ex_Pool.push(self.state_pre, self.action, self.reward, self.state_nex)
               
    def Extract_Par_C(self):
        Flag_p = 0
        for name,param in self.Critic_Net_E.named_parameters():    
            if "weight" in name:
                self.Net_Weight_C[Flag_p] = param.float()
                Flag_p = Flag_p + 1
     
    def Extract_Par_A(self): 
        Flag_p = 0
        for name,param in self.Actor_Net_E.named_parameters():    
            if "weight" in name:
               # print("***",param)
                self.Net_Weight_A[Flag_p] = param.float() 
                Flag_p = Flag_p + 1
    
    def Extract_Par_G_C(self): 
        Flag_p = 0
        for name,param in self.Critic_Net_E.named_parameters():    
            if "weight" in name:
                self.Net_Weight_Grad_C[Flag_p] = param.grad.float()   
                Flag_p = Flag_p + 1
        self.Net_Weight_Grad_C_Set.append(self.Net_Weight_Grad_C)
                      
    def Extract_Par_G_A(self): 
        Flag_p = 0
        for name_1,param_1 in self.Actor_Net_E.named_parameters():    
            if "weight" in name_1:
                self.Net_Weight_Grad_A[Flag_p] = param_1.grad.float()   
                Flag_p = Flag_p + 1
        self.Net_Weight_Grad_A_Set.append(self.Net_Weight_Grad_A)

    def Act_tra(self, act_one):
        act_one = act_one.tolist()
        act_tra = [2 for i in range(State_Dim)]
        for j in act_one:
            act_tra[int(j) - 1] = 20
        act_tra = torch.tensor(act_tra)
        return act_tra

    def act(self, s_t_set, cho, tens, com_num, flag): 
        if tens == 0:
            s_t_set = torch.tensor(s_t_set, dtype=torch.float).unsqueeze(0) 
        a_t_set = []
        act_loss = 0
        
        for s_t in s_t_set: 
            if cho == 1:
                a_t = self.Actor_Net_E(s_t).squeeze(0).detach().numpy()               
            elif cho == 2:
                a_t = self.Actor_Net_T(s_t).squeeze(0).detach().numpy()                 
            a_t = a_t.tolist()

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
            if (random.random() < max([0.2, (1 - com_num * 0.1)])) and (flag == 0):
                return random.choice(Act_Space)
            else:
                return a_t_set[0]
        else:
            return a_t_set

    def Sample_data(self):      
        samples = self.Ex_Pool.sample(Batch_Size)        
        s0, a0, r1, s1 = zip(*samples)        
        s0 = torch.tensor(s0, dtype=torch.float).view(Batch_Size,-1)
        a0 = torch.tensor(a0, dtype=torch.float).view(Batch_Size,-1)
        r1 = torch.tensor(r1, dtype=torch.float).view(Batch_Size,-1)
        s1 = torch.tensor(s1, dtype=torch.float).view(Batch_Size,-1)         
        return s0,a0,r1,s1

    def learn(self):
        
        if len(self.Ex_Pool.memory) < Batch_Size:
            return 0

        s_0, a_0, r_0, s_0_nex = self.Sample_data()
        s_1, a_1, r_1, s_1_nex = self.Sample_data()
        
        def Critic_learn():           
            a_0_nex = self.act(s_0_nex, 2, 1, 1, 1)
            a_0_nex_t = torch.tensor(a_0_nex, dtype=torch.float)
            
            y_true = r_0 + GAMMA * self.Critic_Net_T(s_0_nex, self.Act_Trans(a_0_nex_t), 1).detach()
            y_pred = self.Critic_Net_E(s_0, self.Act_Trans(a_0), 1)
                        
            loss = F.smooth_l1_loss(y_pred, y_true)               
            self.critic_optim.zero_grad()
            loss.backward()                       
            self.critic_optim.step()
            
            #test          
            a_1_nex = self.act(s_1_nex, 2, 1, 1, 1)
            a_1_nex_t = torch.tensor(a_1_nex, dtype=torch.float)
            
            y_true_1 = r_1 + GAMMA * self.Critic_Net_T(s_1_nex, self.Act_Trans(a_1_nex_t), 1).detach()            
            y_pred_1 = self.Critic_Net_E(s_1, self.Act_Trans(a_1), 1)
            loss_1 = F.smooth_l1_loss(y_pred_1, y_true_1) 
            
            self.Q_set.append([s_1.tolist(), self.Act_Trans(a_1).tolist(), y_pred_1.tolist()])          
            self.Loss_Set.append(loss_1.tolist())                     
                                                                                                                                                                        
        def Actor_learn():
            a_1_e = torch.tensor(self.act(s_0, 1, 1, 1, 1), dtype=torch.float)
            Q_mean = torch.mean(-1 * (self.Critic_Net_E(s_0, self.Act_Trans(a_1_e), 1)))
            self.Q_mean.append(float(Q_mean))

            loss = torch.tensor(Q_mean + torch.tensor(self.act_loss))
            self.loss_act_set.append(float(loss))
            loss.requires_grad_(True)
           
            self.actor_optim.zero_grad()
            loss.backward()  
            self.actor_optim.step()
                    
        def Soft_update(net_target, net):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - Tau) + param.data * Tau) 
        
        Critic_learn()
        Actor_learn()
        Soft_update(self.Critic_Net_T, self.Critic_Net_E)
        Soft_update(self.Actor_Net_T, self.Actor_Net_E)
        
#%%
def DBC():
    print(UAV_Num, Beam_Num, Sec_Num, State_Dim, Action_Dim, Sec_list, Act_Space)  
    Comm_Num = 30
    Slot_Num  = 120
    
    Node_list = [(i+1) for i in range(UAV_Num)]  
    Link_Space = list(combinations(Node_list, 2)) 
    
    UAV_List = []
    for e in range(UAV_Num): 
        UAV_List.append(UAV(e+1)) 
    
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
    
    #infile_1 = 'Node_Loc_10.txt'
    #Location_Set(UAV_List,infile_1)
    
    link_per_set = []   
    
    for com_r in range(Comm_Num):
        print("Comm_Round",com_r + 1)
        Link_Set = [] 
        Link_Dis = [] 
        Link_Sec = [] 
        Link_Set_Exi = [] 
        Link_percent = [0 for i in range(Slot_Num)] 
        
        Link_matrix = [[0 for i in range(UAV_Num)] for j in range(UAV_Num)] 
        
        for uav_i in range(UAV_Num):
            UAV_List[uav_i].Par_Init_Class()
            UAV_List[uav_i].move()
                   
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
                    UAV_List[uav_j].slot_num += 1
                    UAV_List[uav_j].Col_Num_Init()
                    act_uav_i = UAV_List[uav_j].act(UAV_List[uav_j].state_pre, 1, 0, com_r, 0)
                    UAV_List[uav_j].action_set.append(act_uav_i)
                    UAV_List[uav_j].action = act_uav_i                          
                    Act_all_node.append(act_uav_i)
    
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
                        
                    col_pro =  (2 ** nei_num_sum - 4) / (2 ** 10 + 200)
    
                    if random.random() >= col_pro:    
                        if Link_Set[link_j] not in Link_Set_Exi:
                            Link_Set_Exi.append(Link_Set[link_j])
    
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
                                           
            for uav_i in range(UAV_Num):
                UAV_List[uav_i].Cal_Beam_Nei()
                UAV_List[uav_i].Save_Data()
                
            if CG.CG(Link_matrix, UAV_Num) == 1:
                print("连通图形成",slot_i)
    
            for uav_p in UAV_List:
                uav_p.learn()
            link_percent_i = (len(Link_Set_Exi)/len(Link_Set)) * 100
            Link_percent[slot_i] = link_percent_i
            link_per_set.append(link_percent_i)
            print("Slot_Num", slot_i + 1, link_percent_i) 

