import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb

ACTION_DIM = 3
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.network(state)


wandb.init(
    # set the wandb project where this run will be logged
    project="LPL-SNN-RL-POC-2",

    # track hyperparameters and run metadata
    config={
        "architecture": "initial",
        "dataset": "rl-memory-path",
    }
)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(1234)
torch.set_printoptions(precision=10, sci_mode=False)

environment_seeds = [4, 1, 2, 3, 4, 5, 6, 7]
env = gym.make('MiniGrid-FourRooms-v0',
               render_mode='human',
               max_steps=50)
env.action_space.seed(42)
state, _ = env.reset(seed=4)
state_dim_1 = state["image"].shape[0]
state_dim_2 = state["image"].shape[1]
state_dim_3 = state["image"].shape[2]
total_state_dim = state_dim_1 * state_dim_2 * state_dim_3

actor = Actor(total_state_dim, ACTION_DIM)
critic = Critic(total_state_dim)

actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

num_episodes = 10000
gamma = 0.99

for episode in range(num_episodes):
    if episode // 25 >= len(environment_seeds):
        environment_seed = random.randint(0, 1000)
    else:
        environment_seed = environment_seeds[episode // 50]

    state, _ = env.reset(seed=environment_seed)
    state = np.reshape(state["image"], [1, total_state_dim])
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_state = next_state["image"]
        next_state = np.reshape(next_state, [1, total_state_dim])
        total_reward += reward

        # Update Critic
        value = critic(state_tensor)
        wandb.log({"value": value})
        next_value = critic(torch.FloatTensor(next_state))
        td_error = reward + (gamma * next_value * (1 - int(done))) - value
        wandb.log({"td_error": td_error})

        critic_loss = td_error ** 2
        wandb.log({"critic_loss": critic_loss})
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Update Actor
        actor_loss = -dist.log_prob(action) * td_error.detach()
        wandb.log({"actor_loss": actor_loss})
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        state = next_state

    print(f'Episode: {episode}, Total Reward: {total_reward}')
