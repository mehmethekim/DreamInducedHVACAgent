import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Any, Sequence
from sinergym.envs.eplus_env import EplusEnv

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent(object):
    def __init__(self, env: EplusEnv,  lr, batch_size, n_actions,
                 gamma=0.99,max_mem_size=1000, epsilon=1.0,eps_end=0.05, eps_dec=5e-4,):
        self.observation_variables = env.get_wrapper_attr('observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')
        self.lower_bounds = np.array([15, 22.5])
        self.higher_bounds = np.array([22.5, 30])
        self.input_size = len(self.observation_variables)
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.range_datacenter =[[15,22.5],[22.5,30]]
        
        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=self.input_size,
                                   fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, self.input_size),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.input_size),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation: List[Any])-> Sequence[Any]:
        obs_dict = dict(zip(self.observation_variables, observation))
        state = T.tensor([obs_dict[var] for var in self.observation_variables], dtype=T.float32)
        
        if np.random.random() > self.epsilon:
            #state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
           
        else:
            action = np.random.choice(self.action_space)
        return self.convert_action(action,observation),action
        

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
    def convert_action(self,action,observation):
        # Mean temp in datacenter zones
        obs_dict = dict(zip(self.observation_variables,observation))
        current_heat_setpoint = obs_dict[
            'west_zone_htg_setpoint']
        current_cool_setpoint = obs_dict[
            'west_zone_clg_setpoint']
        if action == 0:
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif action == 1:
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint - 1
        elif action == 2:
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint
        elif action == 3:
            new_heat_setpoint = current_heat_setpoint - 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif action == 4:
            new_heat_setpoint = current_heat_setpoint - 1
            new_cool_setpoint = current_cool_setpoint - 1
        elif action == 5:
            new_heat_setpoint = current_heat_setpoint - 1
            new_cool_setpoint = current_cool_setpoint 
        elif action == 6:
            new_heat_setpoint = current_heat_setpoint 
            new_cool_setpoint = current_cool_setpoint +1
        elif action == 7:
            new_heat_setpoint = current_heat_setpoint 
            new_cool_setpoint = current_cool_setpoint - 1
        elif action == 8:
            new_heat_setpoint = current_heat_setpoint 
            new_cool_setpoint = current_cool_setpoint
        if new_heat_setpoint < self.range_datacenter[0][0]:
            new_heat_setpoint = self.range_datacenter[0][0]
        elif new_heat_setpoint > self.range_datacenter[0][1]:
            new_heat_setpoint = self.range_datacenter[0][1]
        if new_cool_setpoint < self.range_datacenter[1][0]:
            new_cool_setpoint = self.range_datacenter[1][0]
        elif new_cool_setpoint > self.range_datacenter[1][1]:
            new_cool_setpoint = self.range_datacenter[1][1]
        return (new_heat_setpoint,new_cool_setpoint)