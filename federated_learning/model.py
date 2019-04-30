import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc3 = nn.Linear(fc2_units, action_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
