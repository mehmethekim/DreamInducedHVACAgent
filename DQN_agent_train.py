import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.wrappers import (NormalizeAction,NormalizeObservation)
from utils import initialize_logging
from torch.utils.tensorboard import SummaryWriter
from DQN import Agent
# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
log_dir,model_dir = initialize_logging("DQN") 
writer = SummaryWriter(log_dir=log_dir)

agent = Agent(env=env,lr=0.001, batch_size=64, n_actions=9, eps_end=0.05, eps_dec=5e-4)
num_of_episodes = 100
step = 0
scores, eps_history = [], []
for i in range(num_of_episodes):
    observation, info = env.reset()
    rewards = []
    terminated = False
    current_month = 0
    score = 0
    while not terminated:
        step += 1
        action,raw_action = agent.choose_action(observation)
        observation_, reward, terminated, truncated,info = env.step(action)
        score += reward
        agent.store_transition(observation, raw_action, reward, 
                                observation_, terminated)
        agent.learn()
        observation = observation_
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
        if step % 1000 == 0:
            writer.add_scalar("Mean reward", np.mean(rewards), step)
            writer.add_scalar("Cumulative reward", sum(rewards), step)
    scores.append(score)
    avg_score = np.mean(scores[-100:])
writer.flush()
writer.close()