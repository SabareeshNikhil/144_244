###################################
# Import Required Packages
import torch
import time
import random
import numpy as np
from dqn_agent import Agent
from mlagents.envs import UnityEnvironment


num_episodes=10             


env = UnityEnvironment(file_name=None)

# Get the default brain 
brain_name = env.brain_names[0]

# Assign the default brain as the brain to be controlled
brain = env.brains[brain_name]

# Set the number of actions or action size
action_size = 4

# Set the size of state observations or state size
state_size = brain.vector_observation_space_size


#Initialize Agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# Load trained model weights
agent.network.load_state_dict(torch.load('dqnAgent_Trained_Model.pth'))


# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    # set train mode to false
    env_info = env.reset(train_mode=False)[brain_name]     

    # get initial state of the unity environment 
    state = env_info.vector_observations[0]

    # set the initial episode score to zero.
    score = 0

    # Run the episode loop;
    # At each loop step take an action as a function of the current state observations
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine epsilon-greedy action from current sate
        action = agent.act(state, .01)             

        if round(action) == 0:
            converted_action = np.array([[1,0,0,0]]) # forward
        elif round(action) == 1:
            converted_action = np.array([[2,0,0,0]]) # backward
        elif round(action) == 2:
            converted_action = np.array([[0,0,1,0]]) # counterclock
        elif round(action) == 3:
            converted_action = np.array([[0,0,2,0]]) # clock
        # converted_action = np.column_stack([np.random.randint(0, converted_action_size[i], size=(converted_agent_num)) for i in range(len(converted_action_size))])           
        # converted_action = np.array([[1,0,0,0]])

        # send the action to the environment and receive resultant environment information
        env_info = env.step(converted_action)[brain_name]       

        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished

        # set new state to current state for determining next action
        state = next_state

        # Update episode score
        score += reward

        # If unity indicates that episode is done, 
        # then exit episode loop, to begin new episode
        if done:
            break

    # (Over-) Print current average score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score), end="")


env.close()

# END :) #############

