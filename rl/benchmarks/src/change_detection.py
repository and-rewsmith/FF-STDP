import random

import gymnasium as gym
from torch import nn
from torch import optim
import torch
import wandb
from profilehooks import profile


from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer
from rl.benchmarks.src.change_detection_framework import ChangeDetectionBasic

"""
TODO: 
/ implement actor critic
/ implement the bridge from state to the LPL network, and from LPL network to actor / critic

imitation learn actor
- random offset past correct image show
- somehow we need to detect trials

batch processing
if that doesn't work, implement some sort of search

verify metaplasticity? (not actionable)
"""

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
            # nn.Linear(input_dim, output_dim),
            nn.Linear(input_dim, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE // 2),
            # nn.ReLU(),
            # nn.Linear(HIDDEN_LAYER_SIZE // 2, output_dim),
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
            # nn.Linear(HIDDEN_LAYER_SIZE // 2, 1)
            nn.Linear(HIDDEN_LAYER_SIZE // 2, LAST_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAST_LAYER_SIZE, 1),
        )

    def forward(self, state):
        return self.network(state)


def generate_state_tensor(observation, reward, env):
    # adjust observation to the range [0, n]
    if observation == "blank":
        observation = -1
    observation += 1

    observation_one_hot = [0] * (env.number_of_stimuli + 1)
    observation_one_hot[observation] = 1
    observation_one_hot = torch.Tensor(observation_one_hot)
    reward_one_hot = torch.Tensor([reward])
    state_one_hot = torch.cat((observation_one_hot, reward_one_hot)).unsqueeze(0)
    return state_one_hot


@profile(stdout=False, filename='baseline.prof', skip=True)
def main():
    # Create the ChangeDetectionBasic environment
    # env = ChangeDetectionBasic() # FIX!
    env = ChangeDetectionBasic()

    # Create the LPL Network
    settings = Settings(
        layer_sizes=[100, 100, 100, 100],  # Architecture layer sizes remain unchanged
        data_size=env.number_of_stimuli + 2,  # extra for blank and reward
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
        # set the wandb project where this run will be logged
        project="LPL-SNN-3",

        # track hyperparameters and run metadata
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

    # Set the number of episodes to run
    num_episodes = 5

    # Run the episodes
    for episode in range(num_episodes):
        # Reset the environment for a new episode
        observation = env.reset()
        done = False
        total_reward = 0

        # Run the episode until done
        observation, reward, done, info = env.step(0)  # no lick to start (necessary because obs not provided on reset)
        net.process_data_single_timestep(generate_state_tensor(observation, reward, env))
        network_state = torch.cat(net.layer_activations(), dim=1)
        count = 0
        while not done:
            # Choose an action (0 for no lick, 1 for lick)
            probs = actor(network_state)
            if count % 100 == 0:
                print(probs)
            count += 1
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # Take a step in the environment
            observation, reward, done, info = env.step(action)
            if info['trial_complete']:
                print("New trial started!")
                input()

            # one hot encode observation and reward
            state_one_hot = generate_state_tensor(observation, reward, env)

            # feed into network and retrieve activations
            net.process_data_single_timestep(state_one_hot)
            next_network_state = torch.cat(net.layer_activations(), dim=1)

            # feed into critic
            next_value = critic(next_network_state)
            value = critic(network_state)
            td_error = reward + (GAMMA * next_value * (1 - int(done))) - value

            # critic loss
            critic_loss = td_error ** 2
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            wandb.log({"critic_loss": critic_loss})

            # actor loss
            actor_loss = -dist.log_prob(action) * td_error.detach()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            wandb.log({"actor_loss": actor_loss})

            # Update the total reward for the episode
            total_reward += reward

            # Print the current step information
            print(
                f"Episode: {episode+1}, Time: {info['time']:.2f}, Action: {action}, Observation: {observation}, Reward: {reward}")

            network_state = next_network_state

            wandb.log({"total_reward": total_reward})

        # Print the episode summary
        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Completed Trials: {info['completed_trials']}\n")

        break

    # Close the environment
    env.close()


if __name__ == '__main__':
    main()
