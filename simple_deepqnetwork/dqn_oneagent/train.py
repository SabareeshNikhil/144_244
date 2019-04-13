
"""
DQN for Unity ML-Agents Environments using PyTorch
Includes examples of the following DQN training algorithms:
  -> Vanilla DNQ, 
  -> Double-DQN (DDQN)

The example uses a modified version of the Unity ML-Agents Banana Collection Example Environment.
The environment includes a single agent, who can turn left or right and move forward or backward.
The agent's task is to collect yellow bananas (reward of +1) that are scattered around an square
game area, while avoiding purple bananas (reward of -1). For the version of Bananas employed here,
the environment is considered solved when the average score over the last 100 episodes > 13. 

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

###################################
# Import Required Packages
import torch
import time
import random
import numpy as np
from collections import deque
from dqn_agent import Agent
from mlagents.envs import UnityEnvironment

num_episodes=2000
epsilon=1.0
epsilon_min=0.05
epsilon_decay=0.99
scores = []
scores_average_window = 100      
solved_score = 14                 

# env = UnityEnvironment(file_name="Banana.app")
env = UnityEnvironment(file_name=None)

# Get the default brain 
brain_name = env.brain_names[0]

# Assign the default brain as the brain to be controlled
brain = env.brains[brain_name]


# Set the number of actions or action size
# action_size = len(brain.vector_action_space_size)
action_size = 4

# Set the size of state observations or state size
state_size = brain.vector_observation_space_size

agent = Agent(state_size=state_size, action_size=action_size, dqn_type='DQN', seed=2)


# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    env_info = env.reset(train_mode=True)[brain_name]     

    # get initial state of the unity environment 
    state = env_info.vector_observations[0]

    # set the initial episode score to zero.
    score = 0

    # Run the episode training loop;
    # At each loop step take an epsilon-greedy action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agent network
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    converted_action_size = brain.vector_action_space_size
    converted_agent_num = len(env_info.agents)

    while True:
        # determine epsilon-greedy action from current sate
        action = agent.act(state, epsilon)  

        # if round(action) == 0:
        #     converted_action = np.array([[1,0,0,0]])
        # elif round(action) == 1:
        #     converted_action = np.array([[-1,0,0,0]])
        # elif round(action) == 2:
        #     converted_action = np.array([[0,0,-1,0]])
        # elif round(action) == 3:
        #     converted_action = np.array([[0,0,1,0]])
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

        #Send (S, A, R, S') info to the DQN agent for a neural network update
        agent.step(state, action, reward, next_state, done)

        # set new state to current state for determining next action
        state = next_state

        # Update episode score
        score += reward

        # If unity indicates that episode is done, 
        # then exit episode loop, to begin new episode
        if done:
            break

    # Add episode score to Scores and...
    # Calculate mean score over last 100 episodes 
    # Mean score is calculated over current episodes until i_episode > 100
    scores.append(score)
    average_score = np.mean(scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    # Decrease epsilon for epsilon-greedy policy by decay rate
    # Use max method to make sure epsilon doesn't decrease below epsilon_min
    epsilon = max(epsilon_min, epsilon_decay*epsilon)

    # (Over-) Print current average score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

    # Print average score every scores_average_window episodes
    if i_episode % scores_average_window == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
    
    # Check to see if the task is solved (i.e,. avearge_score > solved_score). 
    # If yes, save the network weights and scores and end training.
    if average_score >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, average_score))

        # Save trained neural network weights
        timestr = time.strftime("%Y%m%d-%H%M%S")
        nn_filename = "dqnAgent_Trained_Model_" + timestr + ".pth"
        torch.save(agent.network.state_dict(), nn_filename)

        # Save the recorded Scores data
        scores_filename = "dqnAgent_scores_" + timestr + ".csv"
        np.savetxt(scores_filename, scores, delimiter=",")
        break


env.close()

# END :) #############

