from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import math

U = 4   # number of UAV
W = 10  # number WN nodes
T = 300     # mission period
K = 4   # subplot
dcov = 20   # E-nodes’ reporting range
BU_max = 4.5*10**5
# battery energy  in range of 2-4 mW·s
Battery_level_W = random.randint(2, 5, size=(W))
Battery_level_W = np.divide(Battery_level_W, 1000)
# collected data from each WN
Collected_WN = np.array([0.0*i for i in range(0, W)])
prev_states = [[[0*i for i in range(0, W)], [0.0*i for i in range(0, W)], [0.0*i for i in range(0, W)],
                [0.0], [0.0], [0.0]] for i in range(0, U)]
BI = 4/1000
BE = 2/1000


# environment setups uav, nodes and reads actions from the model
class Environment:
    def __init__(self):
        self.Qu = np.array([[random.randint(0, 400, size=(2)) if i == 0 else [0.0, 0.0] for i in range(0, T)] for j in range(0, U)])
        self.Qw = random.randint(0, 400, size=[W, 2])
        self.Fw = np.array([[0*i*j for i in range(0, T)] for j in range(0, W)])
        self.Bw = np.array([[0.0*i*j for i in range(0, T)] for j in range(0, W)])
        self.Bu = np.array([[4.5*10**5 if i == 0 else 0.0 for i in range(0, T)] for j in range(0, U)])
        self.CAw = np.array([[0.0 * i * j for i in range(0, T)] for j in range(0, W)])
        self.CW = np.array([[0.0 * i * j for i in range(0, K*T)] for j in range(0, W)])
        self.Muw = np.array([[[0.0*k*j*i for k in range(0, K)] for i in range(0, T)] for j in range(0, W)])

    def harvestEnergyFunc(self, Zu, Gu):
        Psen = -10
        Psat = 7    #dBm
        p = 0
        Pu = 1      #W
        for i in range(0, U):
            p += Zu[i]*Gu[i]*Pu
        if p < Psen:
            eff = 0
        elif Psen <= p < Psat:
            eff = 0 # curve_fitting function
        else:
            eff = 0.557
        return p*eff

    def gain_node(self, qw, qu):    # this function calculates the gain of each E-node
        a = 12.08
        b = 0.11
        alphaL = 3
        alphaN = 5
        G_o = -3    #3dBm
        h = 5   #m

        duw = math.sqrt((qw[0]-qu[0])**2+(qw[1]-qu[1])**2)
        beta = math.asin(h / duw)
        Plos = (1+a*math.exp(-b*(beta-a)))**-1
        Pnlos = 1 - Plos

        return (Plos*G_o*duw)**-alphaL + (Pnlos*G_o*duw)**-alphaN

    # transimmion data size bits/Hz at subslot k
    def M_Inode(self, Duwk, w, D, Gwu, Fw):
        Pw = 0.1/1000
        sigma2 = -90    #dBm
        sslot_tl = 1/K
        numerator = Duwk*Pw*Gwu[w]
        denomenator = 0
        for i in range(0, U):
            for j in range(0, W):
                if j != w and Fw[j] == 1:
                    denomenator += (D[i, j]*Pw*Gwu+sigma2)

        sinr = numerator/denomenator
        return math.log(1+sinr, 2)*sslot_tl

    # energy consumed by the uav in slot t
    def EUav(self, Vu, Duw, Zu):
        v = 1   #s
        Pu = 1  #W
        PI = 0.01 #W
        Pa = 580
        Pb = 134
        V_tip = 200
        e_o = 7.2
        f_o = 0.3
        omega = 1.225
        e_1 = 0.05
        A = 0.79
        P_pro = Pa*(1+3*(Vu/V_tip)**2)+0.5*f_o*omega*e_1*A*Vu**3+Pb*math.sqrt(math.sqrt(1+1/4*(Vu/e_o)**4-(Vu/e_o)**2/2))
        val = 0
        for j in range(0, W):
            for k in range(0, K):
                val += Duw[j,k]*PI*1/K
        return val + P_pro*v + Zu*Pu*v

    # Calculate each UAV observations from the environment
    def sac_observations(self, t):
        global prev_states
        # define Node parameters
        for i in range(0, W):
            # Battery level of each Node and its type
            if t == 0:
                self.Bw[i, 0] = Battery_level_W[i]
            if t != 0 and BE < self.Bw[i, t] < BI:
                self.Fw[i, t] = self.Fw[i, t - 1]
            else:
                if self.Bw[i, t] >= BI:
                    self.Fw[i, t] = 1  # Inode
                if self.Bw[i, t] <= BE:
                    self.Fw[i, t] = 0  # Enode
            # data transmitted by each Node is updated after dqn
        # UAV observations
        states = [
            [[0*i for i in range(0, W)], [0.0*i for i in range(0, W)], [0.0*i for i in range(0, W)], 0.0, 0.0, 0.0]
            for j in range(0, U)]
        for j in range(0, U):
            for i in range(0, W):
                states[j][0][i] = self.Fw[i, t]
                drep = math.sqrt((self.Qu[j, t, 0] - self.Qw[i, 0]) ** 2 + (self.Qu[j, t, 1] - self.Qw[i, 1]) ** 2)
                if drep <= dcov or self.Fw[i, t] == 1:
                    states[j][1][i] = self.Bw[i, t]
                    states[j][2][i] = self.CAw[i, t]
                else:
                    states[j][1][i] = prev_states[j][1][i]
                    states[j][2][i] = prev_states[j][2][i]
            states[j][3] = self.Qu[j, t, 0]
            states[j][4] = self.Qu[j, t, 1]
            states[j][5] = self.Bu[j, t]
        prev_states = states
        return np.concatenate(states)

    #   def sac_actionsPerformed(self, sac_action):
    #   def dqn_actionsPerformed(self, dqn_action):

# Actor neural network
# Critic neural network
# Each user agent has its actor
# Define the V and Q critic for central training of SAC policy
# Define the SAC agent class completely
# Define the DQN at UAV-u and train it locally.
# finish training models


if __name__ == '__main__':
    print(Battery_level_W)
