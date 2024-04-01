import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb

ACTION_DIM = 3
ACTOR_LR = 1e-6
CRITIC_LR = 1e-6

EPISODES_SWITCH_AFTER = 250

HIDDEN_LAYER_SIZE = 2048
LAST_LAYER_SIZE = 64

TIMESTEPS = 80

NUM_STORED_STATES = 10


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

# environment_seeds = [4, 383, 428, 410, 147, 63, 698, 911, 919, 135, 904, 109, 146, 1, 178, 16, 189, 337]
# environment_seeds = [4, 146, 1, 919, 178, 4, 16, 904]
environment_seeds = [146]
env = gym.make('MiniGrid-FourRooms-v0',
               render_mode='human',
               max_steps=TIMESTEPS)
env.action_space.seed(42)
state, _ = env.reset(seed=4)
state_dim_1 = state["image"].shape[0]
state_dim_2 = state["image"].shape[1]
state_dim_3 = state["image"].shape[2]
total_state_dim = NUM_STORED_STATES * (VISION_SIZE * VISION_SIZE *
                                       (NUM_OBJECTS + NUM_COLORS + NUM_STATES) + NUM_DIRECTIONS) + TIMESTEPS

actor = Actor(total_state_dim, ACTION_DIM)
critic = Critic(total_state_dim)

actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

num_episodes = 100000
gamma = 0.99

successes = 0
failures = 0

states = []
for episode in range(num_episodes):
    if episode // EPISODES_SWITCH_AFTER >= len(environment_seeds):
        environment_seed = random.randint(0, 1000)
        print(f"Episode: {episode}, Seed: {environment_seed}")
    else:
        environment_seed = environment_seeds[episode // EPISODES_SWITCH_AFTER]

    # random_seed_from_seeds = random.randint(0, len(environment_seeds) - 1)
    # environment_seed = environment_seeds[random_seed_from_seeds]
    # environment_seed = 4

    print(f"Episode: {episode}, Seed: {environment_seed}")

    state, _ = env.reset(seed=environment_seed)
    vision = state["image"]
    direction = state["direction"]
    state = convert_observation_to_spike_input(vision, direction)

    states.append(state)
    if len(states) > NUM_STORED_STATES:
        states.pop(0)
    while len(states) < NUM_STORED_STATES:
        states.append(state)

    timestep = 0
    total_state_tensor = torch.cat(states, dim=1)
    timestep_tensor = torch.zeros(TIMESTEPS)
    timestep_tensor[timestep-1] = 1
    timestep_tensor = timestep_tensor.unsqueeze(0)
    total_state_tensor = torch.cat((total_state_tensor, timestep_tensor), dim=1)

    done = False
    total_reward = 0

    prev_actor_loss = None  # TODOPRE: remove
    prev_prob_move = None  # TODOPRE: remove
    old_state = None  # TODOPRE: remove
    while not done:
        total_state_tensor = torch.FloatTensor(total_state_tensor)
        probs = actor(total_state_tensor)
        print(f"probs: {probs}")

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # print(f"action: {action}")

        # # TODOPRE: remove
        # # if probs is higher relative to old probs, and loss was neg, debug
        # if prev_actor_loss is not None and prev_prob_move is not None:
        #     # if torch.equal(total_state_tensor[:, 0:-2], old_state[:, 0:-2]) and prev_actor_loss < 0 and probs[0][action].item() - prev_prob_move < 0:
        #     if torch.equal(total_state_tensor, old_state) and prev_actor_loss < 0 and probs[0][action].item() - prev_prob_move > 0:
        #         input()

        old_state = total_state_tensor  # TODOPRE: remove
        observation, reward, terminated, truncated, _ = env.step(action.item())

        # TODOPRE: try this with increased reward the closer you are
        if 8 in observation["image"]:
            reward += 0.005

        if terminated:
            successes += 1
        if truncated:
            failures += 1

        done = terminated or truncated
        vision = observation["image"]
        direction = observation["direction"]
        next_single_state = convert_observation_to_spike_input(vision, direction)
        total_reward += reward

        states.append(next_single_state)
        if len(states) > NUM_STORED_STATES:
            states.pop(0)

        timestep += 1
        total_state_tensor_next = torch.cat(states, dim=1)
        timestep_tensor = torch.zeros(TIMESTEPS)
        timestep_tensor[timestep-1] = 1
        timestep_tensor = timestep_tensor.unsqueeze(0)
        total_state_tensor_next = torch.cat((total_state_tensor_next, timestep_tensor), dim=1)
        # total_state_tensor_next = torch.cat((total_state_tensor_next, torch.Tensor([[timestep]])), dim=1)

        # if episode < EPISODES_SWITCH_AFTER and not is_in_bounds(env.agent_pos):
        #     done = True

        # Update Critic
        value = critic(total_state_tensor)
        wandb.log({"value": value})
        next_value = critic(torch.FloatTensor(total_state_tensor_next))
        td_error = reward + (gamma * next_value * (1 - int(done))) - value
        # print(f"reward: {reward}")
        # print(f"next_value: {next_value}")
        # print(f"value: {value}")
        # print(f"td_error: {td_error}")
        wandb.log({"td_error": td_error})

        critic_loss = td_error ** 2
        wandb.log({"critic_loss": critic_loss})
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Update Actor
        actor_loss = -dist.log_prob(action) * td_error.detach()
        prev_actor_loss = actor_loss.item()
        wandb.log({"actor_loss": actor_loss})
        # print(f"actor_loss: {actor_loss}")
        # print()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        total_state_tensor = total_state_tensor_next

        wandb.log({"successes": successes})
        wandb.log({"failures": failures})

        # TODOPRE: remove
        prev_prob_move = probs[0][action].item()
        prev_actor_loss = actor_loss.item()

    print(f'Episode: {episode}, Total Reward: {total_reward}')
