import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)


torch.autograd.set_detect_anomaly(True)
torch.manual_seed(1234)
torch.set_printoptions(precision=10, sci_mode=False)

env = gym.make('MiniGrid-FourRooms-v0',
               render_mode='human',
               max_steps=20)
env.action_space.seed(42)
state, _ = env.reset(seed=4)
state_dim_1 = state["image"].shape[0]
state_dim_2 = state["image"].shape[1]
state_dim_3 = state["image"].shape[2]
total_state_dim = state_dim_1 * state_dim_2 * state_dim_3
action_dim = env.action_space.n

actor = Actor(total_state_dim, action_dim)
critic = Critic(total_state_dim)

actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    state, _ = env.reset(seed=4)
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
        next_value = critic(torch.FloatTensor(next_state))
        td_error = reward + (gamma * next_value * (1 - int(done))) - value

        critic_loss = td_error ** 2
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Update Actor
        actor_loss = -dist.log_prob(action) * td_error.detach()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        state = next_state

    print(f'Episode: {episode}, Total Reward: {total_reward}')
