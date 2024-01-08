import gymnasium as gym
from DQN import Agent
import numpy as np

env = gym.make('CartPole-v1')
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
                  input_dims=[4], lr=0.001)
scores, eps_history = [], []
n_games = 10
    
for i in range(n_games):
    score = 0
    done = False
    rewards = []
    observation,info = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, truncated,info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, 
                                observation_, done)
        agent.learn()
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)

# Model path should be the name of the model plus the time stamp which can be obtanied from the system
# Example: "model_2020-07-16_19-42-00.pth"
model_path = "model_2020-07-16_19-42-00.pth"
# Save the model
agent.save_model(model_path)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Progress')
plt.show()

# Run the environment with the model in human mode
env = gym.make('CartPole-v1', render_mode='human')
agent.load_model(model_path)

observation = env.reset()
done = False
while not done:
    env.render()
    action = agent.choose_action(observation)
    observation, reward, done,truncated, info = env.step(action)
env.close()



env.close()