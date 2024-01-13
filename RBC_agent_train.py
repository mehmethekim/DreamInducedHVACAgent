import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.wrappers import (NormalizeAction,NormalizeObservation)
from utils import initialize_logging
from torch.utils.tensorboard import SummaryWriter

from sinergym.utils.controllers import RBC5Zone
# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
#Initialize logging for tensorboard and model saving
log_dir,model_dir = initialize_logging("RBC")
writer = SummaryWriter(log_dir=log_dir)
agent = RBC5Zone(env)
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
        action = agent.act(obs)
        obs_, reward, terminated,truncated, info = env.step(action)
        score += reward
        obs = obs_
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