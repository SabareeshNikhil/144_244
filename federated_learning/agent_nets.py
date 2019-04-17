import torch

import numpy as np
import random
import time
from dqn_agent import Agent
from model import QNetwork

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():

    def __init__(self, state_size, action_size, num_agents, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99,
    	learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):

        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)

        self.num_agents = num_agents

        # Initialize agents
        self.agents = [Agent(state_size=state_size, action_size=action_size, dqn_type='DQN', seed=time.time() + i) for i in range(num_agents)]

        # Most recently averaged net
        self.last_average_net = QNetwork(state_size, action_size, seed=time.time() + self.num_agents).to(device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, actions, rewards, next_state, dones):
        for i in range(self.num_agents):
            # print("STATES and State:", state, state[i])
            self.agents[i].step(state[i], actions[i], rewards[i], next_state[i], dones[i])

    def act(self, state, eps):
        arr = []
        for i in range(self.num_agents):
            a = self.agents[i].act(state[i], eps)
            # print("ARR and a:", arr, a)
            arr.append(a)
        return np.array(arr)

    def download_global_net(self, global_net):
        for i in range(self.num_agents):
            a = self.agents[i].download_global_net(global_net)

    def get_average_network(self):
        # zero average net
        for average_param in self.last_average_net.parameters():
            average_param.data.copy_(average_param.data * 0)

        # get sum of params of all agent networks
        for i in range(self.num_agents):
            net_i = self.agents[i].network
            for average_param, i_param in zip(self.last_average_net.parameters(), net_i.parameters()):
                average_param.data.copy_(average_param.data + i_param.data)

        # divide to get average
        for average_param in self.last_average_net.parameters():
                average_param.data.copy_(average_param.data/self.num_agents)

        return self.last_average_net