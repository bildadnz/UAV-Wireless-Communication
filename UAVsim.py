import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import math
import threading
import random
from torch.nn import functional as F

U = 4   # number of UAV
W = 10  # number WN nodes
T = 300     # mission period
K = 4   # subplot
dcov = 20   # E-nodes’ reporting range
BU_max = 4.5*10**5  # UAV max energy
BU_min = 3*10**4    # UAV min energy
EPS = 500
# battery energy  in range of 2-4 mW·s
Battery_level_W = np.random.randint(2, 5, size=W)
Battery_level_W = np.divide(Battery_level_W, 1000)
# collected data from each WN
Collected_WN = np.array([0.0*i for i in range(0,0, W)])
prev_states = [[0] * (3 * W + 3) for _ in range(0,U)]
BI = 4/1000
BE = 2/1000


# environment setups uav, nodes and reads actions from the model
class Environment:
    def __init__(self):
        self.Qu = np.array([[np.random.randint(0, 400, size=2) if i == 0 else [0.0, 0.0] for i in range(0, T)] for j in range(0, U)])
        self.Qw = np.random.randint(0, 400, size=[W, 2])
        self.Fw = np.array([[0*i*j for i in range(0, T)] for j in range(0, W)])
        self.Bw = np.array([[0.0*i*j for i in range(0, T)] for j in range(0, W)])
        self.Bu = np.array([[BU_max if i == 0 else 0.0 for i in range(0, T)] for j in range(0, U)])
        self.CAw = np.array([[0.0 * i * j for i in range(0, T)] for j in range(0, W)])
        self.CW = np.array([[0.0*t*j for t in range(0, T)] for j in range(0, W)])
        self.Muw = np.array([[[[0.0*k*j*i*u for u in range(0, U)] for j in range(0, W)] for k in range(0, K)]
                             for i in range(0, T)])
        self.Guw = np.array([[[0*i*j*t for j in range(0, U)] for i in range(0, W)] for t in range(0,T)])
        self.ZUt = np.array([[0*j*i for i in range(0, U)] for j in range(0, T)])
        self.DUWt = np.array([[[[0*j*t*i*k for i in range(0, U)] for j in range(0, W)]for k in range(0, K)]
                              for t in range(0, T)])
        self.Hw = np.array([[0*t*i for i in range(0,W)] for t in range(0,T)])

    def _energyTransfer_eff(self, p):
        Psen = -10
        Psat = 7
        popt = [-1.33074721e+02, - 2.33336935e+01, - 1.60367485e+00, - 5.42036659e-02, - 9.00624077e-04,
                - 5.88089624e-06]
        if p < Psen:
            eff = 0
        elif Psen <= p < Psat:
            # curve_fitting function
            eff = popt[0] + popt[1] * (p - 30) + popt[2] * (p - 30) ** 2 + popt[3] * (p - 30) ** 3 + popt[4] * (
                    p - 30) ** 4 + popt[5] * (p - 30) ** 5
        else:
            eff = popt[0] + popt[1] * (7 - 30) + popt[2] * (7 - 30) ** 2 + popt[3] * (7 - 30)** 3 + popt[4]*(
                    7- 30)**4 + popt[5] * (7 - 30) ** 5
        return eff

    def _harvestEnergyFunc(self, Zu, Gu):
        p = 0
        pp = 0
        Pu = 1      #W
        for i in range(0, U):
            pp += Zu[i]*Gu[i]*Pu
            p += Zu[i]*10**((Gu[i]-30)/10)*Pu
        return p*self._energyTransfer_eff(pp)

    def _gain_node(self, qw, qu):    # this function calculates the gain of each E-node
        a = 12.08
        b = 0.11
        alphaL = 3
        alphaN = 5
        G_o = 0.5  # 0.4 dBm
        h = 5   #m

        duw = math.sqrt((qw[0]-qu[0])**2+(qw[1]-qu[1])**2)
        beta = 0
        if (h / duw) <= 1:
            beta = math.asin(h / duw)
        Plos = (1+a*math.exp(-b*(beta-a)))**-1
        Pnlos = 1 - Plos
        val = (Plos*G_o*duw)**-alphaL + (Pnlos*G_o*duw)**-alphaN
        if val > 20:
            val = random.randint(10, 20)
        return val

    # transmission data size bits/Hz at sub slot k
    def _M_Inode(self, Dact, w, D, Gwu, ti):
        Pw = 0.1/1000
        sigma2 = 10**((-90-30)/10) #watts
        sslot_tl = 1/K
        numerator = Dact*Pw*Gwu[w]
        denomenator = 0
        for i in range(0, U):
            for j in range(0, W):
                if j != w and self.Fw[j, ti] == 1:
                    denomenator += (D[j, i]*Pw*Gwu[j]+sigma2)
        c = 0
        if denomenator != 0:
            sinr = numerator/denomenator
            c = math.log(1 + sinr, 2) * sslot_tl
        return c

    # energy consumed by the uav in slot t
    def _EUav(self, Vu, Zu, u, t):
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
        # propulsion power
        P_pro = Pa*(1+3*(Vu/V_tip)**2)+0.5*f_o*omega*e_1*A*Vu**3+Pb*math.sqrt(math.sqrt(1+1/4*(Vu/e_o)**4-(Vu/e_o)**2/2))
        val = 0
        for j in range(0, W):
            for k in range(0, K):
                val += self.DUWt[t, k, j, u]*PI*1/K
        # total power consumed is communication, propulsion and WET energy consumption of UAV
        return val + P_pro*v + Zu*Pu*v

    def rewards_SACAction(self, t):
        Exp = (BI - BE) / T
        # A metric HoE is used to measure each WN’s time-varying energy demands
        if t > 0:
            self.Hw[t] = np.array([0 if self.Fw[w, t] == 1 else self.Hw[t - 1, w] + 1 if
            self._harvestEnergyFunc(self.ZUt[t - 1], self.Guw[t - 1, w]) < Exp else max(self.Hw[t - 1, w] - 1, 1) for w
                                   in range(W)])
        else:
            self.Hw[t] = np.array([0 if self.Fw[w, t] == 1 else 1 for w in range(0, W)])
        # a weight for charging all e-nodes by a uav
        Nu = np.array([0*u for u in range(0, U)])
        Fac = np.array([0 * u for u in range(0, U)])
        Pu = 1  # W
        # reward  Denote UAV-u’s reward on WET to all the E-nodes in slot t given as
        Bu_Wet = [0*u for u in range(0, U)]
        for i in range(0, U):
            for j in range(0, W):
                if self.Fw[j, t] == 0 and t+1 < T:
                    p = self.ZUt[t, i] * self.Guw[t, j, i] * Pu
                    if self.Bw[j, t+1]-self.Bw[j, t] != 0:
                        Nu[i] += (self._energyTransfer_eff(p)*p)/(self.Bw[j, t+1]-self.Bw[j, t])
                        Fac[i] += (self.Bw[j, t+1]-self.Bw[j, t])*self.Hw[t, j]
            Bu_Wet[i] = Nu[i]*Fac[i]*20
        # reward on WDC decision from inodes
        Cmin = 100  # bits/Hz at time slot t
        Bu_Wdc = [np.sum(np.array([self.CAw[i, t] - t/T*Cmin for i in range(0, W)]))*0.01 for u in range(0, U)]
        # reward on UAV battery energy saving
        n = t + 1
        if t + 1 == T:
            n = t
        Bu_Es = [(self.Bu[u, n]-BU_min)*(10**-6) for u in range(0, U)]
        # reward on UAV distance safe distance
        dmin = 2    #m
        Bu_Sd = [0*u for u in range(0, U)]
        rewards = []
        for u in range(U):
            for j in range(U):
                if j != u:
                    dist = np.linalg.norm(self.Qu[j, t] - self.Qu[u, t])
                    if dist < dmin:
                        Bu_Sd[u] = -1
                        break
            rewards.append([Bu_Wet[u] + Bu_Wdc[u] + Bu_Es[u] + Bu_Sd[u]])
        return rewards

    def rewards_DQNAction(self, t, k):
        Cmin = 100  # bits/Hz at time slot t
        return [np.sum(np.array([self.Muw[t, k, i, u] for i in range(W)]))
                        + np.sum(np.array([self.CAw[i, t] - (t-1)/T*Cmin for i in range(W)])) for u in range(U)]

    # Calculate each UAV observations from the environment
    def sac_observations(self, t):
        global prev_states
        # define Node parameters
        for i in range(W):
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
        states = []
        for j in range(U):
            fw = [self.Fw[i, t] for i in range(W)]
            bw = []
            caw = []
            for i in range(W):
                drep = math.sqrt((self.Qu[j, t, 0] - self.Qw[i, 0]) ** 2 + (self.Qu[j, t, 1] - self.Qw[i, 1]) ** 2)
                if drep <= dcov or self.Fw[i, t] == 1:
                    bw.append(self.Bw[i, t])
                    caw.append(self.CAw[i, t])
                else:
                    bw.append(prev_states[j][W + i])
                    caw.append(prev_states[j][2 * W + i])
            pos_x = self.Qu[j, t, 0]
            pos_y = self.Qu[j, t, 1]
            bu = self.Bu[j, t]

            # Flatten: [Fw..., Bw..., CAw..., pos_x, pos_y, Bu]
            state = fw + bw + caw + [pos_x, pos_y, bu]
            states.append(state)

        prev_states = states
        return np.array(states, dtype=np.float32)

    def sac_actionsPerformed(self, sac_action, t):
        #  WET decision to transfer energy to E-node is given by sac action Z
        # add sac_action to array list
        Vu = [t.item() for t in sac_action[1]]
        ang = [t.item() for t in sac_action[0]]
        Zu = [1 if t.item() > 0 else 0 for t in sac_action[2]]

        if t+1 < T:
            for u in range(0, U):
                self.Qu[u, t+1, 0] = self.Qu[u, t, 0] + Vu[u]*math.cos(ang[u])
                self.Qu[u, t+1, 1] = self.Qu[u, t, 1] + Vu[u]*math.sin(ang[u])
                self.ZUt[t, u] = Zu[u]
            Pw = 0.1/1000
            # update battery levels of UAV and Wireless Nodes

            for i in range(W):
                for j in range(U):
                    val = self._gain_node(self.Qw[i], self.Qu[j, t])
                    self.Guw[t, i, j] = val

            for i in range(0, W):
                if self.Fw[i, t] == 0:  # Enode
                    val = self.Bw[i, t] + self._harvestEnergyFunc(self.ZUt[t], self.Guw[t, i])
                    if np.isnan(val):
                        val = 4/1000
                    self.Bw[i, t+1] = min(4/1000, val)
                else:   # Inode
                    Inode_energy = 0
                    for j in range(0, U):
                        for k in range(0, K):
                            Inode_energy += self.DUWt[t, k, i, j]*Pw
                    self.Bw[i, t+1] = max(self.Bw[i, t]-Inode_energy, 0)
            for i in range(U):   # UAV
                val = self.Bu[i, t]-self._EUav(Vu[i], self.ZUt[t, i], i, t)
                if np.isnan(val):
                    val = 0
                self.Bu[i, t+1] = max(val, 0)

    def dqn_Observations(self, p, t):
        states = []
        for j in range(0,U):
            state = [self.Qu[j, t, 0], self.Qu[j, t, 1]] + [self.CAw[i, t] for i in range(0, W)] + [p]
            states.append(state)
        return states

    def dqn_actionsPerformed(self, k, t, actions, Wca):
        for u in range(U):
            Dact = 0
            maxw = torch.argmax(actions[u]).item()
            if self.Fw[maxw, t] == 1 and Wca[maxw] != -1:
                Dact = 1
            self.DUWt[t, k, maxw, u] = Dact
            Duw = self.DUWt[t, k]
            GUw = [self.Guw[t, i, u] for i in range(0, W)]
            muw = self._M_Inode(Dact, maxw, Duw, GUw, t)
            self.Muw[t, k, maxw, u] = muw

        for i in range(W):
            for j in range(U):
                for ki in range(K):
                    self.CW[i, t] += self.Muw[t, ki, i, j]
            for ts in range(0, t):
                self.CAw[i, ts] += self.CW[i, ts]


# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, max_size=131072):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch_size = min(128, len(self.buffer))
        return random.sample(self.buffer, 1)


# Actor neural network
# Each user agent has its actor
class Actor(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3*W+3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 3)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        out = self.fc5(x)
        xang = [self.tanh(out[u][0]) for u in range(U)]
        Vu = [self.tanh(out[u][1]) for u in range(U)]
        Zu = [self.tanh(out[u][2]) for u in range(U)]
        return xang, Vu, Zu


# Critic neural network
class V_Critic(nn.Module):
    def __init__(self, hidden_dim=256):
        super(V_Critic, self).__init__()
        self.fc1 = nn.Linear((3*W+3)*U, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        out = self.fc5(x)
        return out


class Q_Critic(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Q_Critic, self).__init__()
        self.fc1 = nn.Linear((3*W+3)*U+3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


# DQN policy neural network
class E_DQN(nn.Module):
    def __init__(self, hidden_dim=256):
        super(E_DQN, self).__init__()
        self.fc1 = nn.Linear(W+3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, W)

    def forward(self, obsver):
        x = torch.relu(self.fc1(obsver))
        x2 = torch.relu(self.fc2(x))
        x3 = torch.relu(self.fc3(x2))
        x4 = torch.relu(self.fc4(x3))
        out = torch.relu(self.fc5(x4))
        return out


# Define the V and Q critic losses for central training of SAC policy
class SAC_Losses:
    def __init__(self, actions, state, reward, statePlus):
        self.state = state
        self.actions = actions
        self.reward = reward
        self.statePlus = statePlus

    def Vc1_Loss(self, alphaU, Qmin, Vc1_out):
        # Log-probabilities of sampled actions
        mean = torch.mean(self.actions)
        log_std = torch.log(torch.std(self.actions))
        dist = torch.distributions.Normal(mean, log_std)
        target = torch.mean(self.actions)*(Qmin-dist.entropy()*alphaU)
        p1 = 0.5*(Vc1_out-target)**2
        return torch.mean(p1)

    def Qc_Loss(self, gamma, Qc, Vc2_out):
        target = self.reward + gamma * torch.mean(Vc2_out)
        target = target.reshape(Qc.shape[0], 1)
        criterion = nn.MSELoss()
        loss = criterion(Qc, target)
        return loss

    def Act_loss(self, alphaU, QminNoisy):
        noise = torch.normal(mean=0.0, std=0.1, size=self.actions.shape)  # ε ∼ 𝒩(0, σ²)
        noisy_action = self.actions + noise
        mean = torch.mean(noisy_action)
        log_std = torch.log(torch.std(noisy_action))
        dist = torch.distributions.Normal(mean, log_std)
        return torch.mean(dist.entropy()*alphaU-QminNoisy)

    def AlphaU_loss(self, alphaU, Hbar):
        return torch.mean(self.actions)*(-alphaU*torch.log10(torch.mean(self.actions))-alphaU*Hbar)


# Define the SAC agent class completely
class SAC_Agent:
    def __init__(self, lr=0.0003, gamma=0.99, tau=0.999, alpha=0.0002):
        self.gamma = gamma
        self.tau = tau
        self.Actor = Actor()
        self.V_critic1 = V_Critic()
        self.V_critic2 = V_Critic()
        self.Q_critic1 = Q_Critic()
        self.Q_critic2 = Q_Critic()

        self.path = "src/models/"

        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr=lr)
        self.Vcritic1_optimizer = optim.Adam(self.V_critic1.parameters(), lr=lr)
        self.Qcritic1_optimizer = optim.Adam(self.Q_critic1.parameters(), lr=lr)
        self.Qcritic2_optimizer = optim.Adam(self.Q_critic2.parameters(), lr=lr)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.alphaU_optimizer = optim.Adam([self.alpha], lr=lr)

    def saveData(self):
        torch.save({
            'Actor_state_dict': self.Actor.state_dict(),
            'V1_state_dict': self.V_critic1.state_dict(),
            'V2_state_dict': self.V_critic2.state_dict(),
            'Q1_state_dict': self.Q_critic1.state_dict(),
            'Q2_state_dict': self.Q_critic2.state_dict()
        }, self.path+"model.pt")

    def loadData(self):
        checkpoint = torch.load(self.path+"model.pt")
        self.Actor.load_state_dict(checkpoint['Actor_state_dict'])
        self.Actor.eval()
        self.V_critic1.load_state_dict(checkpoint['V1_state_dict'])
        self.V_critic1.eval()
        self.V_critic2.load_state_dict(checkpoint['V2_state_dict'])
        self.V_critic2.eval()
        self.Q_critic1.load_state_dict(checkpoint['Q1_state_dict'])
        self.Q_critic1.eval()
        self.Q_critic2.load_state_dict(checkpoint['Q2_state_dict'])
        self.Q_critic2.eval()

    def getSacAction(self, Observation_U):
        return self.Actor(Observation_U)

    def update(self, mini_batch):
        states, actions, rewards, next_states = zip(*mini_batch)
        xang, Vu, Zu = zip(*actions)
        action_z = [[xang[0][j], Vu[0][j], Zu[0][j]]for j in range(len(xang[0]))]

        states = torch.stack(states)
        actions = torch.tensor(action_z, dtype=torch.float32)
        rewards = torch.tensor(rewards[0], dtype=torch.float32)
        next_states = torch.stack(next_states)
        noise = torch.normal(mean=0.0, std=0.1, size=actions.shape)  # ε ∼ 𝒩(0, σ²)
        noisy_action = actions + noise

        states = states.reshape(states.size()[1], states.size()[2])

        #states = F.normalize(states, dim=1)
        #actions = F.normalize(actions, dim=1)
        #rewards = F.normalize(rewards, dim=1)
        #next_states = F.normalize(next_states, dim=1)

        loss_func = SAC_Losses(states, actions, rewards, next_states)

        vc1 = self.V_critic1(states)

        qu1 = self.Q_critic1(states, actions)
        qu2 = self.Q_critic2(states, actions)
        qmin1 = torch.min(qu1, qu2)

        self.Vcritic1_optimizer.zero_grad()
        lossVc1 = loss_func.Vc1_Loss(self.alpha, qmin1, vc1)
        lossVc1.backward()
        self.Vcritic1_optimizer.step()

        # soft update Vcritic 2 parameters
        with torch.no_grad():
            for target_param, param in zip(self.V_critic2.parameters(), self.V_critic1.parameters()):
                target_param.data = self.tau * target_param.data + (1 - self.tau) * param

        # update q critic network load new q1 and q2 tensors for back propagation
        qu1 = self.Q_critic1(states, actions)
        vc2 = self.V_critic2(next_states)

        self.Qcritic1_optimizer.zero_grad()
        lossQc1 = loss_func.Qc_Loss(self.gamma, qu1, vc2)
        lossQc1.backward()
        self.Qcritic1_optimizer.step()

        qu2 = self.Q_critic2(states, actions)
        vc2 = self.V_critic2(next_states)

        self.Qcritic2_optimizer.zero_grad()
        lossQc2 = loss_func.Qc_Loss(self.gamma, qu2, vc2)
        lossQc2.backward()
        self.Qcritic2_optimizer.step()

        # add noise to avoid over fitting and update weights for the actor network
        qu1N = self.Q_critic1(states, noisy_action)
        qu2N = self.Q_critic2(states, noisy_action)
        qminNois = torch.min(qu1N, qu2N)

        self.actor_optimizer.zero_grad()
        lossAct = loss_func.Act_loss(self.alpha, qminNois)
        lossAct.backward()
        self.actor_optimizer.step()

        self.alphaU_optimizer.zero_grad()
        lossAU = loss_func.AlphaU_loss(self.alpha, 2)
        lossAU.backward()


# Define the DQN at UAV-u and train it locally.
class DQN_Agent:
    def __init__(self, gamma=0.99):
        self.lr_start = 0.01
        self.lr_end = 0.000001
        self.erS = 0.9
        self.erE = 0.02
        self.gamma = gamma
        self.Q_1 = E_DQN()
        self.Q_T = E_DQN()
        self.buffer = ReplayBuffer()

        self.Q_1_optimizer = optim.Adam(self.Q_1.parameters(), lr=self.lr_start)
        self.sched_learn = torch.optim.lr_scheduler.ExponentialLR(self.Q_1_optimizer,
                                                                  gamma=(self.lr_end/self.lr_start)**(1/EPS))

    def get_actions(self, ObservationU):
        return self.Q_1(ObservationU)

    def soft_update_target(self, epi, tau=0.999):
        with torch.no_grad():
            if random.randint(1, 200)/100 < max(self.erE, self.erS - (self.erS - self.erE) * epi / EPS):
                for target_param, local_param in zip(self.Q_T.parameters(), self.Q_1.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update(self, mini_batch, epi):
        observs_z, actions_z, rewards_z, next_observs_z = zip(*mini_batch)
        # print(type(next_observs_z[0]))
        observs = observs_z[0]  # shape: (batch_size, obs_dim)
        actions = actions_z[0]
        rewards = rewards_z[0]
        next_observs = next_observs_z[0]

        # match new actions to old
        acts_new = self.get_actions(observs)
        are_equal = torch.eq(actions, acts_new)
        # Count how many elements differ
        num_different = torch.sum(~are_equal).item()

        if num_different == 0:
            pass

        acts_new = F.normalize(acts_new, dim=0)
        rewards = F.normalize(rewards, dim=0)

        with torch.no_grad():
            next_acts = self.Q_T(next_observs)
            next_new = F.normalize(next_acts, dim=0)
            max_next_acts = torch.max(next_new)

        self.Q_1_optimizer.zero_grad()
        p1 = torch.mul(self.gamma, max_next_acts)
        target = torch.add(p1, rewards)
        maxt = torch.max(acts_new)
        crit = nn.MSELoss()
        loss = crit(maxt, target)
        # print(maxt, loss)
        torch.autograd.set_detect_anomaly(True)
        loss.backward()

        self.Q_1_optimizer.step()
        self.soft_update_target(epi)


def run_uav_agents_async(uavs, Od_Uk):
    actions_results = [None] * U

    def worker(u):
        actions_results[u] = uavs[u].get_actions(Od_Uk[u])

    threads = []
    for u in range(U):
        thread = threading.Thread(target=worker, args=(u,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    return actions_results


def Trainer():
    Ctotal = np.array([0*j for j in range(EPS)])
    Sac = SAC_Agent()
    Sac.loadData()
    uavs = [DQN_Agent() for _ in range(U)]
    buffer_sac = ReplayBuffer()
    for i in range(EPS):
        # Initialize the locations and battery levels of all UAVs and WNs;
        Env = Environment()
        # Initialize all te observations
        Os_Ut = torch.tensor(Env.sac_observations(0), dtype=torch.float32)
        states = torch.flatten(Os_Ut)
        states = states.repeat(U, 1)
        Od_Uk = [torch.tensor(Env.dqn_Observations(k, 0), dtype=torch.float32) for k in range(K)]
        for t in range(T-1):
            # actions performed and reward obtained
            As_Ut = Sac.getSacAction(Os_Ut)
            Env.sac_actionsPerformed(As_Ut, t)
            reward_sac = Env.rewards_SACAction(t)
            acts = []
            reward_dqn = []
            Wca = [w for w in range(W)]
            for k in range(K):
                # actions at k
                acts.append(run_uav_agents_async(uavs, Od_Uk[k]))
                Env.dqn_actionsPerformed(k, t, acts[k], Wca)
                reward_dqn.append(Env.rewards_DQNAction(t, k))
            # next states and experience
            next_Os_Ut = torch.tensor(Env.sac_observations(t + 1), dtype=torch.float32)
            nxt_states = torch.flatten(next_Os_Ut)
            nxt_states = nxt_states.repeat(U, 1)

            experience = (states, As_Ut, reward_sac, nxt_states)
            buffer_sac.add(experience)
            next_Od_Uk = []
            for k in range(K):
                # next observations and experience
                next_Od_Uk.append(torch.tensor(Env.dqn_Observations(k, t+1), dtype=torch.float32))
                for ui in range(U):
                    experi = (Od_Uk[k][ui], acts[k][ui], torch.tensor(reward_dqn[k][ui], dtype=torch.float32), next_Od_Uk[k][ui])
                    uavs[ui].buffer.add(experi)
            # print(len(buffer_sac.buffer))
            # print(reward_sac)
            # update dqn network
            for agent in uavs:
                batch = agent.buffer.sample()
                agent.update(batch, i)

            # update sac network
            Sac.update(buffer_sac.sample())
            Od_Uk = next_Od_Uk
            Os_Ut = next_Os_Ut

        print(np.sum(Env.Muw))
        Ctotal[i] = np.sum(Env.CW)
        print(Ctotal[i])
        for agent in uavs:
            agent.sched_learn.step()
    Sac.saveData()


# finish training models
if __name__ == '__main__':
    print(Battery_level_W)
    Trainer()
