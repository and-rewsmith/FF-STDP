import gymnasium as gym
import numpy as np
import wandb
import torch
from tqdm import tqdm
from torch import nn
from torchviz import make_dot

ENCODE_SPIKE_TRAINS = False

BATCH_SIZE = 1

"""
Useful links:
https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/fourrooms.py
https://github.com/Farama-Foundation/Minigrid/blob/df4e6752af069f77f0537d700d283f4c02dd4e35/minigrid/core/constants.py
https://github.com/Farama-Foundation/Minigrid/blob/df4e6752af069f77f0537d700d283f4c02dd4e35/minigrid/core/world_object.py#L273
"""

# encoding for each tile of visibility
VISIBILITY_ENCODING_LEN = 3
NUM_OBJECTS = 11
NUM_COLORS = 6
NUM_STATES = 3

NUM_ACTIONS = 3
NUM_DIRECTIONS = 4

# reward is a float that is one hot encoded
NUM_REWARD = 1

MAP_LENGTH_X = 19
MAP_LENGTH_Y = 19

INIT_TIMESTEPS = 1000
TRAIN_TIMESTEPS = 100000
INFERENCE_TIMESTEPS = 500

ACTOR_LR = 1e-7
CRITIC_LR = 1e-6
STATE_PREDICTOR_LR = 1e-5

VISION_SIZE = 7


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.optim = torch.optim.Adam(self.parameters(), lr=ACTOR_LR)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.optim = torch.optim.Adam(self.parameters(), lr=CRITIC_LR)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class StatePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.optim = torch.optim.Adam(self.parameters(), lr=STATE_PREDICTOR_LR)

    def forward(self, x):
        return self.linear(x)


class DecoderGroup:
    def __init__(self, input_dim, action_dim):
        self.actor = Actor(input_dim, 20, action_dim)
        self.critic = Critic(input_dim, 20, 1)
        self.state_predictor = StatePredictor(input_dim, input_dim)
        self.prev_state = None
        self.prev_value = None
        self.prev_actor_neg_log_prob = None

    def train(self, state, resulting_state, action, reward):
        # Compute RPE
        value = self.critic(state)
        if self.prev_value is not None:
            # print(f"value: {value.item()}")
            # print(f"rpe: {reward + value.item() - self.prev_value}")
            # print(f"reward: {reward}")
            # print()
            wandb.log({"rpe": reward + value.item() - self.prev_value})
            wandb.log({"value": value.item()})
            rpe = reward + value - self.prev_value
        else:
            rpe = reward

        # Train Actor
        if self.prev_actor_neg_log_prob is not None:
            actor_loss = self.prev_actor_neg_log_prob * -1 * rpe.clone().detach()
            wandb.log({"actor_loss": actor_loss.item()})
            self.actor.optim.zero_grad()
            actor_loss.backward()  # need retain because we are using rpe in two places
            self.actor.optim.step()

        # Train Critic
        if self.prev_value is not None:
            critic_loss = rpe ** 2
            wandb.log({"critic_loss": critic_loss.item()})
            self.critic.optim.zero_grad()
            critic_loss.backward()
            # graph = make_dot(critic_loss, params=dict(self.critic.named_parameters()))
            # graph.render("computation_graph", format="png")
            # graph.view()
            # input()
            self.critic.optim.step()

        # Train StatePredictor
        if self.prev_state is not None:
            predicted_state = self.state_predictor(self.prev_state)
            state_predictor_loss = torch.nn.functional.mse_loss(predicted_state, state)
            wandb.log({"state_predictor_loss": state_predictor_loss.item()})
            self.state_predictor.optim.zero_grad()
            state_predictor_loss.backward()
            self.state_predictor.optim.step()

        # Update previous state and value
        self.prev_state = state.detach()  # TODOPRE: prob not needed
        self.prev_value = value.detach()

        # update prev actor neg log prob
        action_prob = self.actor(resulting_state)
        self.prev_actor_neg_log_prob = -torch.log(action_prob[0][action])

        return torch.multinomial(action_prob, 1).item()

    def forward(self, state):
        action_prob = self.actor(state)
        return torch.multinomial(action_prob, 1).item()


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
    return agent_pos[0] >= 6 and agent_pos[0] < 9 and agent_pos[1] >= 11 and agent_pos[1] < 12


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)
    torch.set_printoptions(precision=10, sci_mode=False)

    wandb.init(
        # set the wandb project where this run will be logged
        project="LPL-SNN-RL-POC",

        # track hyperparameters and run metadata
        config={
            "architecture": "initial",
            "dataset": "rl-memory-path",
        }
    )

    decoder_group_input_size = VISION_SIZE * VISION_SIZE * \
        (NUM_OBJECTS + NUM_COLORS + NUM_STATES) + NUM_DIRECTIONS
    decoder_group = DecoderGroup(
        decoder_group_input_size,
        NUM_ACTIONS)

    env = gym.make(
        'MiniGrid-FourRooms-v0',
        render_mode='none',
        max_steps=1000000)
    env.action_space.seed(42)

    observation, info = env.reset(seed=4)

    # make action on environment
    action = env.action_space.sample(
        np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int8))

    # TODOPRE: train with RPE
    successes = 0
    failures = 0
    for _ in tqdm(range(TRAIN_TIMESTEPS), desc="collect for decoding"):
        wandb.log({"successes": successes, "failures": failures})

        old_observation = observation

        # some actions aren't used
        if action > 2:
            print("UNEXPECTED ACTION")
            exit()

        observation, reward, terminated, truncated, info = env.step(action)
        if not is_in_bounds(env.agent_pos):
            failures += 1
            reward = -1

        if reward != 0:
            # print("reward: ", reward)
            reward = reward / 20

        # convert to spike encoding based on: action, reward, visibility, and direction
        # feed this into the network
        visibility = old_observation["image"]
        direction = old_observation["direction"]
        spike_encoding = convert_observation_to_spike_input(
            visibility, direction)

        visibility = observation["image"]
        direction = observation["direction"]
        resulting_state_spike_encoding = convert_observation_to_spike_input(
            visibility, direction)

        action = decoder_group.train(spike_encoding, resulting_state_spike_encoding, action, reward)

        if terminated or truncated or not is_in_bounds(env.agent_pos):
            # print("RESETTING")
            if terminated:
                # print("terminated: ", terminated)
                successes += 1
            # if truncated:
            #     print("truncated: ", truncated)
            # if not is_in_bounds(env.agent_pos):
            #     print("agent_pos: ", env.agent_pos)
            observation, info = env.reset(seed=4)

    env.close()
