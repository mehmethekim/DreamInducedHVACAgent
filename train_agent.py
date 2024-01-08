import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.wrappers import (NormalizeAction,NormalizeObservation)
from utils import initialize_logging
from torch.utils.tensorboard import SummaryWriter
from custom_controllers.DQN_controller import Agent
from sinergym.utils.controllers import RBCDatacenter
# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-datacenter-hot-continuous-stochastic-v1')
 
#Initialize logging for tensorboard and model saving
log_dir,model_dir = initialize_logging("Random")
writer = SummaryWriter(log_dir=log_dir)
# create DQN controller
agent = Agent( env = env, n_actions=9, eps_end=0.01, lr=0.001,batch_size=64)
#agent= DDPGController(env)
agent = RBCDatacenter(env)
num_of_episodes = 10
step = 0
scores, eps_history = [], []
for i in range(num_of_episodes):
    obs, info = env.reset()
    rewards = []
    terminated = False
    current_month = 0
    score = 0
    while not terminated:
       
        step += 1
        #action = env.action_space.sample() # From the observation, generate an action
        #action = agent.act(obs) # From the observation, generate an action
        #print("The action type is ",type(action),action)
        # action,q_value = agent.choose_action(obs)
        # obs_, reward, terminated,truncated, info = env.step(action)
        # score += reward
        # agent.store_transition(obs, q_value, reward,
        #                         obs_, terminated)
        # agent.learn()
        # obs = obs_
        action = env.action_space.sample()#agent.act(obs)
        obs_, reward, terminated,truncated, info = env.step(action)
        score += reward
        obs = obs_
        # obs_next, reward, terminated, truncated, info = env.step(action)
        # agent.update_ddpg_network(obs, action, reward, obs_next, terminated)
 
        # obs = obs_next
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
        if step % 1000 == 0:
            writer.add_scalar("Mean reward", np.mean(rewards), step)
            writer.add_scalar("Cumulative reward", sum(rewards), step)
    scores.append(score)
 
    avg_score = np.mean(scores[-100:])
 
    # print('episode ', i, 'score %.2f' % score,
    #         'average score %.2f' % avg_score,
    #         'epsilon %.2f' % agent.epsilon)
writer.flush()
writer.close()