import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from replay_memory import ReplayBuffer

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GlobalNet():

    def __init__(self, state_size, action_size, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99,
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

        self.network = QNetwork(state_size, action_size, seed).to(device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def show_weights(self):
        for param in self.network.parameters():
            print(param.data.shape)
    
    # def update_weights(self, avg_model, alpha):
    #     # for param in self.network.parameters():
    #         # param.data.copy_(param.data == 1)

    #     # for network_param, avg_param in zip(self.network.parameters(), avg_model.parameters()):
    #     #     network_param.data.copy_(alpha*avg_param.data + (1.0-alpha)*network_param.data)
    #     # self.show_weights()

    def receive_upload(self, average_net):
        for global_param, avg_param in zip(self.network.parameters(), average_net.parameters()):
            global_param.data.copy_(avg_param.data)
            # global_param.data.copy_(alpha*avg_param.data + (1.0-alpha)*global_param.data)
        
