import random
import warnings

import gymnasium as gym
from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
import wandb
from profilehooks import profile

from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer
from rl.benchmarks.src.change_detection_framework2 import ChangeDetectionBasic

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.init")

ACTION_DIM = 2
ACTOR_LR = 1e-6 * 5
CRITIC_LR = 1e-6 * 5

HIDDEN_LAYER_SIZE = 2048
LAST_LAYER_SIZE = 32

BATCH_SIZE = 1

ENCODE_SPIKE_TRAINS = False

GAMMA = 0.99


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE // 2, LAST_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAST_LAYER_SIZE, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE // 2, LAST_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAST_LAYER_SIZE, 1),
        )

    def forward(self, state):
        return self.network(state)


def generate_state_tensor(observation, reward, env):
    observation_one_hot = torch.zeros(BATCH_SIZE, env.number_of_stimuli + 1)
    observation_long = observation.long() + 1
    observation_one_hot[range(BATCH_SIZE), observation_long] = 1
    reward_tensor = reward.clone().detach().unsqueeze(1)
    state_one_hot = torch.cat((observation_one_hot, reward_tensor), dim=1)
    return state_one_hot


if __name__ == '__main__':
    # Create the ChangeDetectionBasic environment
    env = ChangeDetectionBasic(batch_size=BATCH_SIZE)

    # Create the LPL Network
    settings = Settings(
        layer_sizes=[100, 100, 100, 100],
        data_size=env.number_of_stimuli + 2,
        batch_size=BATCH_SIZE,
        learning_rate=0.01,
        epochs=10,
        encode_spike_trains=ENCODE_SPIKE_TRAINS,
        dt=.01,
        percentage_inhibitory=50,
        exc_to_inhib_conn_c=0.25,
        exc_to_inhib_conn_sigma_squared=60,
        layer_sparsity=0.9,
        decay_beta=0.85,
        tau_mean=1200,
        tau_var=0.02,
        tau_stdp=0.1,
        tau_rise_alpha=0.005,
        tau_fall_alpha=.05,
        tau_rise_epsilon=0.002,
        tau_fall_epsilon=0.02,
        device=torch.device("cpu")
    )

    wandb.init(
        project="LPL-SNN-3",
        config={
            "architecture": "initial",
            "dataset": "rl-change-detection",
            "settings": settings,
        }
    )

    net = Net(settings).to(device=settings.device)
    visualizer = NetworkVisualizer(net)

    actor_input_dim = sum(settings.layer_sizes)
    actor = Actor(actor_input_dim, ACTION_DIM).to(device=settings.device)

    critic_input_dim = sum(settings.layer_sizes)
    critic = Critic(critic_input_dim).to(device=settings.device)

    actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    num_episodes = 1

    total_reward = torch.zeros(BATCH_SIZE)
    for episode in range(num_episodes):
        observation = env.reset()
        done = torch.zeros(BATCH_SIZE, dtype=torch.bool)

        observation, reward, done, info = env.step(torch.zeros(BATCH_SIZE, dtype=torch.long))
        net.process_data_single_timestep(generate_state_tensor(observation, reward, env))
        network_state = torch.cat(net.layer_activations(), dim=1)
        original_observation = observation
        should_lick = torch.zeros(BATCH_SIZE, dtype=torch.bool)
        episode_failed = torch.ones(BATCH_SIZE, dtype=torch.bool)
        rewarded = torch.zeros(BATCH_SIZE, dtype=torch.bool)

        while not done:
            probs = actor(network_state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            if episode % 2 == 0:
                optimal_action = torch.zeros(BATCH_SIZE, dtype=torch.long)
                optimal_action.logical_or_(should_lick)
                optimal_action.logical_and_(rewarded.logical_not())
                random_sample = torch.rand(BATCH_SIZE)
                optimal_action.logical_and_(random_sample < 0.5)
                action = optimal_action

            observation, reward, done, info = env.step(action)
            print(f"time: {env.time}, observation: {observation[0]}, action: {action[0]}, reward: {reward[0]}")
            print()

            rewarded.logical_or_(reward == 1)

            if reward[0] == 1:
                input()

            # TODOPRE: we need to do something with this, maybe useful for stopped samples if we decide to stop them
            # original_observation = torch.where(done, observation, original_observation)

            episode_failed = torch.where(reward == 1, False, episode_failed)
            should_lick = torch.logical_and(observation != -1, observation != original_observation)
            print(should_lick)

            state_one_hot = generate_state_tensor(observation, reward, env)
            net.process_data_single_timestep(state_one_hot)
            next_network_state = torch.cat(net.layer_activations(), dim=1)

            next_value = critic(next_network_state)
            value = critic(network_state)
            done_int = 1 if done else 0
            td_error = reward + (GAMMA * next_value * (1 - done)) - value

            critic_loss = td_error ** 2
            critic_optim.zero_grad()
            critic_loss.mean().backward()
            critic_optim.step()
            wandb.log({"critic_loss": critic_loss.mean()})

            if episode % 2 == 0:
                # print(f"optimal action: {should_lick[0]}, episode failed: {episode_failed[0]}")
                optimal_action = optimal_action & episode_failed
                non_optimal_action = ~optimal_action
                optimal_action_probs = torch.zeros_like(probs)
                optimal_action_prob = 0.5
                complement_prob = 1 - optimal_action_prob
                optimal_action_probs[range(BATCH_SIZE), optimal_action.long()] = optimal_action_prob
                optimal_action_probs[range(BATCH_SIZE), non_optimal_action.long()] = complement_prob
                actor_loss = F.mse_loss(probs, optimal_action_probs)
            else:
                actor_loss = -dist.log_prob(action) * td_error.detach()

            actor_optim.zero_grad()
            actor_loss.mean().backward()
            actor_optim.step()
            wandb.log({"actor_loss": actor_loss.mean()})

            total_reward += reward
            network_state = next_network_state

            wandb.log({"total_reward": total_reward.mean()})

        print(f"Episode: {episode+1}, Average Total Reward: {total_reward.mean()}\n")

    env.close()
