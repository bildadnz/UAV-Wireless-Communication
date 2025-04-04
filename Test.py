import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# ================== Environment Parameters ==================
U = 4   # Number of UAVs
W = 10  # Number of WN nodes
BE = np.random.randint(2, 5, size=W)  # Battery energy levels (mW·s)
T = 300  # Mission period (time steps)
K = 4   # Subplot division
AREA = 400 * 400  # Area size
DCOV = 20  # E-nodes’ reporting range


class UAV_WPCN_Env:
    def __init__(self):
        self.num_uavs = U
        self.num_wns = W
        self.max_steps = T
        self.reset()

    def reset(self):
        self.t = 0
        self.uav_positions = np.random.rand(self.num_uavs, 2) * np.sqrt(AREA)  # UAV initial positions
        self.wn_battery = np.random.randint(2, 5, size=self.num_wns)  # WNs' battery levels
        self.wn_data = np.random.rand(self.num_wns) * 100  # WNs' data amount
        return self.get_state()

    def get_state(self):
        return np.concatenate([self.uav_positions.flatten(), self.wn_battery, self.wn_data])

    def step(self, action_sac, action_dqn):
        # Move UAVs based on SAC actions
        self.uav_positions += action_sac.reshape(self.num_uavs, 2) * 0.1

        # Update WNs' battery based on UAV energy transmission
        self.wn_battery = self.wn_battery.astype(float)
        self.wn_battery += np.random.rand(self.num_wns) * 2

        # Wireless data collection based on DQN policy
        collected_data = action_dqn * (self.wn_battery > 2)  # Only transmit if charged
        reward = np.sum(collected_data)  # Total collected data as reward
        self.wn_data = np.maximum(0, self.wn_data - collected_data)

        self.t += 1
        done = self.t >= self.max_steps
        return self.get_state(), reward, done, {}

# ================== SAC for Tier 1 (WEN Policy) ==================
class SAC_Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().numpy().flatten()

# ================== DQN for Tier 2 (WDC Policy) ==================
class DQN_Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.q_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=10000)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.q_network(state).detach().numpy().flatten() > 0  # Binary transmission decision

    def update(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        actions = torch.argmax(actions, dim=1)
        q_values = self.q_network(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ================== Training HDRL ==================
def train_hdrl(env, sac_agent, dqn_agent, num_episodes=1000):
    C_total_values = []

    for episode in range(num_episodes):
        state = env.reset()
        C_total = 0

        for t in range(env.max_steps):
            action_sac = sac_agent.select_action(state)  # SAC for UAV movement & WEN
            action_dqn = dqn_agent.select_action(state)  # DQN for WDC decision

            next_state, reward, done, _ = env.step(action_sac, action_dqn)
            C_total += reward

            # Store experience in DQN buffer
            dqn_agent.replay_buffer.append((state, action_dqn, reward, next_state))

            if done:
                break

            state = next_state

            # Train DQN
            dqn_agent.update()

        C_total_values.append(C_total)
        print(f"Episode {episode+1}: C_total = {C_total}")

    return C_total_values


if __name__ == '__main__':
    # ================== Run Training and Plot Results ==================
    env = UAV_WPCN_Env()
    sac_agent = SAC_Agent(state_dim=env.get_state().shape[0], action_dim=env.num_uavs * 2)
    dqn_agent = DQN_Agent(state_dim=env.get_state().shape[0], action_dim=env.num_wns)

    # Train HDRL Agents
    C_total_trained = train_hdrl(env, sac_agent, dqn_agent)
    # Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(C_total_trained, label="C_total", color='b', linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Total Collected Data (C_total)")
    plt.title("C_total vs. Training Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()
