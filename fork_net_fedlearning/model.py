import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Variables
        self.shared_frac = 0.9637
        self.local_frac = 1 - self.shared_frac

        self.shared_state_size = round(state_size * self.shared_frac)
        self.local_state_size = state_size - self.shared_state_size

        self.shared_fc1_size = round(fc1_units * self.shared_frac)
        self.shared_fc2_size = round(fc2_units * self.shared_frac)

        self.local_fc1_size = round(fc1_units * self.local_frac)
        self.local_fc2_size = round(fc2_units * self.local_frac)

        # Layers
        self.shared1 = nn.Linear(self.shared_state_size, self.shared_fc1_size)
        self.shared2 = nn.Linear(self.shared_fc1_size, self.shared_fc2_size)

        self.local1 = nn.Linear(self.local_state_size, self.local_fc1_size)
        self.local2 = nn.Linear(self.local_fc1_size, self.local_fc2_size)

        self.fc3 = nn.Linear(self.shared_fc2_size + self.local_fc2_size, action_size)

        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        shared_state = state[:,:self.shared_state_size]
        local_state = state[:,self.shared_state_size:]

        sharedX = F.relu(self.shared1(shared_state))
        sharedX = F.relu(self.shared2(sharedX))

        localX = F.relu(self.local1(local_state))
        localX = F.relu(self.local2(localX))

        combinedX = torch.cat([sharedX,localX], dim=1)

        return self.fc3(combinedX)

        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)
