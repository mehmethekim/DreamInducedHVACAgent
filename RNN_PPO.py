from typing import Tuple, Type
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class RNN_PPO_Network(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        rnn_hidden_size: int = 64,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(RNN_PPO_Network, self).__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # RNN layer
        self.rnn = nn.LSTMCell(feature_dim, rnn_hidden_size)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, last_layer_dim_pi), nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor, rnn_states: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        # features shape: (batch_size, feature_dim)
        # rnn_states: Tuple containing hidden state and cell state for the LSTM
        rnn_states = self.rnn(features, rnn_states)

        # Extract hidden state for policy and value networks
        rnn_hidden_state, _ = rnn_states

        # Policy network
        latent_policy = self.policy_net(rnn_hidden_state)

        # Value network
        latent_value = self.value_net(rnn_hidden_state)

        return latent_policy, latent_value, rnn_states


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Tuple[int] = (64, 64),
        activation_fn: Type[nn.Module] = nn.Tanh,
        rnn_hidden_size: int = 64,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        self.ortho_init = False
        self.rnn_hidden_size = rnn_hidden_size

    def make_actor_critic_network(
        self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
    ) -> RNN_PPO_Network:
        return RNN_PPO_Network(
            self.features_dim,
            rnn_hidden_size=self.rnn_hidden_size,
            **self.net_arch,
        )
