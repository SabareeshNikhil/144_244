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

class Agent():

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

        """
        # DQN Agent Q-Network
        # For DQN training, two neural network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """
        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, actions, rewards, next_state, dones):
        # Save experience in replay memory
        for i in range(len(actions)):
            # print("Step ACTIONS", actions, actions[i], state[i])
            self.memory.add(state[i], actions[i], rewards[i], next_state[i], dones[i])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)


    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        num_agents = len(action_values[0])

        # print("AGENT ACT VALUES", action_values,  np.argmax(action_values.cpu().data.numpy()[0], 1),  np.array([random.choice(np.arange(self.action_size)) for i in range(num_agents)]))
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()[0], 1)
        else:
            return np.array(np.array([random.choice(np.arange(self.action_size)) for i in range(num_agents)]))
            


    # Update value parameters using given batch of experience tuples.
    def learn(self, experiences, gamma, DQN=True):
        
        """
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states).gather(1, actions)

        if (self.dqn_type == 'DDQN'):
        #Double DQN
        #************************
            Qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(1)
            Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions].unsqueeze(1)

        else:
        #Regular (Vanilla) DQN
        #************************
            # Get max Q values for (s',a') from target model
            Qsa_prime_target_values = self.target_network(next_states).detach()
            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)        
        
        # Compute Q targets for current states 
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        
        # Compute loss (error)
        loss = F.mse_loss(Qsa, Qsa_targets)

        # print(Qsa, Qsa_targets)
        # print(loss)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)

    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
