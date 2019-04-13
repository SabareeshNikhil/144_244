
import torch
import time
import random
import numpy as np
from collections import deque
from dqn_agent import Agent
from mlagents.envs import UnityEnvironment

def convert_action(a):
    if round(a) == 0:
        converted_action = np.array([[1,0,0,0]]) # forward
    elif round(a) == 1:
        converted_action = np.array([[2,0,0,0]]) # backward
    elif round(a) == 2:
        converted_action = np.array([[0,0,1,0]]) # counterclock
    elif round(a) == 3:
        converted_action = np.array([[0,0,2,0]]) # clock
    
    return converted_action

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

agent = Agent(state_size=state_size, action_size=action_size, dqn_type='DQN', seed=4)

# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    env_info = env.reset(train_mode=True)[brain_name]     

    # get initial state of the unity environment 
    state = env_info.vector_observations

    num_agents = len(env_info.agents)

    # set the initial episode scores to zero.
    scores = np.float32([0] * num_agents)

    # Run the episode training loop;
    # At each loop step take an epsilon-greedy action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agent network
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 

    while True:
        # determine epsilon-greedy action from current sate
        actions = agent.act(state, epsilon)  

        converted_actions = [convert_action(a) for a in actions]
        # print ("CONVERTED_ACTIONS", actions, converted_actions)

        # send the action to the environment and receive resultant environment information
        env_info = env.step(converted_actions)[brain_name]        

        next_state = env_info.vector_observations   # get the next state
        rewards = env_info.rewards                   # get the reward
        dones = env_info.local_done                  # see if episode has finished

        #Send (S, A, R, S') info to the DQN agent for a neural network update
        agent.step(state, actions, rewards, next_state, dones)

        # set new state to current state for determining next action
        state = next_state

        # Update episode score
        scores += rewards

        # If unity indicates that episode is done, 
        # then exit episode loop, to begin new episode
        if all(dones):
            break

    print ("Scores", scores)

    # # Add episode score to Scores and...
    # # Calculate mean score over last 100 episodes 
    # # Mean score is calculated over current episodes until i_episode > 100
    # scores.append(score)
    # average_score = np.mean(scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    # # Decrease epsilon for epsilon-greedy policy by decay rate
    # # Use max method to make sure epsilon doesn't decrease below epsilon_min
    epsilon = max(epsilon_min, epsilon_decay*epsilon)

    # # (Over-) Print current average score
    # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

    # # Print average score every scores_average_window episodes
    # if i_episode % scores_average_window == 0:
    #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
    
    # # Check to see if the task is solved (i.e,. avearge_score > solved_score). 
    # # If yes, save the network weights and scores and end training.
    # if average_score >= solved_score:
    #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, average_score))

    #     # Save trained neural network weights
    #     timestr = time.strftime("%Y%m%d-%H%M%S")
    #     nn_filename = "dqnAgent_Trained_Model_" + timestr + ".pth"
    #     torch.save(agent.network.state_dict(), nn_filename)

    #     # Save the recorded Scores data
    #     scores_filename = "dqnAgent_scores_" + timestr + ".csv"
    #     np.savetxt(scores_filename, scores, delimiter=",")
    #     break


env.close()


