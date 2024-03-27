import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb

ACTION_DIM = 3
ACTOR_LR = 1e-4
CRITIC_LR = 1e-5 * 5

EPISODES_SWITCH_AFTER = 300


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

# encoding for each tile of visibility
VISIBILITY_ENCODING_LEN = 3
NUM_OBJECTS = 11
NUM_COLORS = 6
NUM_STATES = 3

NUM_ACTIONS = 3
NUM_DIRECTIONS = 4

VISION_SIZE = 7


def convert_observation_to_spike_input(
        vision: np.ndarray, direction: int):
    # Define the encoding dimensions
    encoding_size = NUM_OBJECTS + NUM_COLORS + NUM_STATES

    # Initialize the binary encoded array
    binary_array = np.zeros((VISION_SIZE, VISION_SIZE, encoding_size))

    # Iterate over each cell in the observation array
    for i in range(VISION_SIZE):
        for j in range(VISION_SIZE):
            obj_idx, color_idx, state_idx = vision[i, j]

            binary_array[i, j, obj_idx] = 1
            binary_array[i, j, NUM_OBJECTS + color_idx] = 1
            binary_array[i, j, NUM_OBJECTS + NUM_COLORS + state_idx] = 1

    # Collapse the binary_array into a 1D tensor
    feature_dim = VISION_SIZE * VISION_SIZE * encoding_size

    # One-hot encode the direction
    direction_one_hot = np.zeros(4)
    direction_one_hot[direction] = 1

    # Concatenate the collapsed tensor with direction, action, and reward
    # encodings
    collapsed_tensor = torch.from_numpy(
        binary_array.reshape(1, feature_dim)).float()
    direction_tensor = torch.from_numpy(direction_one_hot).float().unsqueeze(0)
    final_tensor = torch.cat(
        (collapsed_tensor,
         direction_tensor),
        dim=1)

    return final_tensor


def is_in_bounds(agent_pos):
    return agent_pos[0] >= 6 and agent_pos[0] < 9 and agent_pos[1] >= 10 and agent_pos[1] < 13


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
total_state_dim = VISION_SIZE * VISION_SIZE * \
    (NUM_OBJECTS + NUM_COLORS + NUM_STATES) + NUM_DIRECTIONS

actor = Actor(total_state_dim, ACTION_DIM)
critic = Critic(total_state_dim)

actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

num_episodes = 10000
gamma = 0.99

successes = 0
failures = 0

for episode in range(num_episodes):
    if episode // EPISODES_SWITCH_AFTER >= len(environment_seeds):
        environment_seed = random.randint(0, 1000)
    else:
        environment_seed = environment_seeds[episode // EPISODES_SWITCH_AFTER]

    state, _ = env.reset(seed=environment_seed)
    vision = state["image"]
    direction = state["direction"]
    state = convert_observation_to_spike_input(vision, direction)

    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        if terminated:
            successes += 1
        if truncated:
            failures += 1

        done = terminated or truncated
        vision = next_state["image"]
        direction = next_state["direction"]
        next_state = convert_observation_to_spike_input(vision, direction)
        total_reward += reward

        if episode < EPISODES_SWITCH_AFTER and not is_in_bounds(env.agent_pos):
            done = True

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

        wandb.log({"successes": successes})
        wandb.log({"failures": failures})

    print(f'Episode: {episode}, Total Reward: {total_reward}')
