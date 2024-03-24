import gymnasium as gym
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.src.constants import DT, EXC_TO_INHIB_CONN_C, EXC_TO_INHIB_CONN_SIGMA_SQUARED, LAYER_SPARSITY, LEARNING_RATE, PERCENTAGE_INHIBITORY, TAU_FALL_ALPHA, TAU_FALL_EPSILON, TAU_MEAN, TAU_RISE_ALPHA, TAU_RISE_EPSILON, TAU_STDP, TAU_VAR
from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer

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

INIT_TIMESTEPS = 100
TRAIN_TIMESTEPS = 10000
INFERENCE_TIMESTEPS = 500


def convert_observation_to_spike_input(
        vision: np.ndarray, direction: int, action: int, reward: float):
    # Define the encoding dimensions
    encoding_size = NUM_OBJECTS + NUM_COLORS + NUM_STATES

    # Initialize the binary encoded array
    binary_array = np.zeros((7, 7, encoding_size))

    # Iterate over each cell in the observation array
    for i in range(7):
        for j in range(7):
            obj_idx, color_idx, state_idx = vision[i, j]
            binary_array[i, j, obj_idx] = 1
            binary_array[i, j, NUM_OBJECTS + color_idx] = 1
            binary_array[i, j, NUM_OBJECTS + NUM_COLORS + state_idx] = 1

    # Collapse the binary_array into a 1D tensor
    feature_dim = 7 * 7 * encoding_size

    # One-hot encode the direction
    direction_one_hot = np.zeros(4)
    direction_one_hot[direction] = 1

    # One-hot encode the action
    action_one_hot = np.zeros(NUM_ACTIONS)
    action_one_hot[action] = 1

    # Binary encode the reward
    # TODO: change to incorporate reward decay
    reward_binary = np.array([1]) if reward > 0 else np.array([0])

    # Concatenate the collapsed tensor with direction, action, and reward
    # encodings
    collapsed_tensor = torch.from_numpy(
        binary_array.reshape(1, feature_dim)).float()
    direction_tensor = torch.from_numpy(direction_one_hot).float().unsqueeze(0)
    action_tensor = torch.from_numpy(action_one_hot).float().unsqueeze(0)
    reward_tensor = torch.from_numpy(reward_binary).float().unsqueeze(0)
    final_tensor = torch.cat(
        (collapsed_tensor,
         direction_tensor,
         action_tensor,
         reward_tensor),
        dim=1)

    return final_tensor


def train_decode_from_training_run(activations_per_timestep, positions_x, positions_y,
                                   rewards, past_actions, num_epochs=10000, learning_rate=0.00005, batch_size=256):
    # Create a DataLoader for batching the data
    dataset = TensorDataset(
        activations_per_timestep,
        positions_x,
        positions_y,
        rewards,
        past_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the decoder model
    input_size = activations_per_timestep.shape[1]
    hidden_size = 128
    num_decoder_layers = 2
    decoder = Decoder(
        input_size,
        hidden_size,
        num_decoder_layers,
        NUM_ACTIONS,
        MAP_LENGTH_X,
        MAP_LENGTH_Y)

    # Define the loss functions and optimizer
    position_x_criterion = nn.MSELoss()
    position_y_criterion = nn.MSELoss()
    reward_criterion = nn.MSELoss()
    action_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch_activations, batch_positions_x, batch_positions_y, batch_rewards, batch_actions = batch

            # Forward pass
            predicted_actions, predicted_x_positions, predicted_y_positions, predicted_rewards = decoder(
                batch_activations)

            # Compute the losses
            # TODO: confirm that operator can be used on multidim
            position_x_loss = position_x_criterion(
                predicted_x_positions, batch_positions_x)
            position_y_loss = position_y_criterion(
                predicted_y_positions, batch_positions_y)
            reward_loss = reward_criterion(predicted_rewards, batch_rewards)
            action_loss = action_criterion(predicted_actions, batch_actions)
            total_loss = position_x_loss + position_y_loss + reward_loss + action_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

    return decoder


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_actions, num_positions_x, num_positions_y):
        super(Decoder, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.action_output = nn.Linear(hidden_size, num_actions)
        self.position_output_x = nn.Linear(hidden_size, num_positions_x)
        self.position_output_y = nn.Linear(hidden_size, num_positions_y)
        self.reward_output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        action_probs = self.softmax(self.action_output(x))
        position_probs_x = self.softmax(self.position_output_x(x))
        position_probs_y = self.softmax(self.position_output_y(x))
        reward = self.reward_output(x)
        return action_probs, position_probs_x, position_probs_y, reward


def smooth_for_decoding(activations, position_x,
                        position_y, rewards, past_actions):
    # Collapse activations from all layers into a single dimension for each
    # timestep
    collapsed_activations = []
    for t in range(0, len(activations)):
        activations = torch.cat([layer.squeeze()
                                for layer in activations_per_timestep[t]])
        collapsed_activations.append(activations)
    collapsed_activations = torch.stack(collapsed_activations)

    # One-hot encode position_x
    position_x_one_hot = np.zeros((len(position_x), MAP_LENGTH_X))
    position_x_one_hot[np.arange(len(position_x)), position_x] = 1
    position_x_one_hot = torch.tensor(position_x_one_hot, dtype=torch.float32)

    # One-hot encode position_y
    position_y_one_hot = np.zeros((len(position_y), MAP_LENGTH_Y))
    position_y_one_hot[np.arange(len(position_y)), position_y] = 1
    position_y_one_hot = torch.tensor(position_y_one_hot, dtype=torch.float32)

    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

    # One-hot encode past actions
    action_one_hot = np.zeros((len(past_actions), NUM_ACTIONS))
    action_one_hot[np.arange(len(past_actions)), past_actions] = 1
    action_one_hot = torch.tensor(action_one_hot, dtype=torch.float32)

    return collapsed_activations, position_x_one_hot, position_y_one_hot, rewards, action_one_hot


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)
    torch.set_printoptions(precision=10, sci_mode=False)

    settings = Settings(
        layer_sizes=[50, 50, 50, 50],
        data_size=7 * 7 * (NUM_COLORS + NUM_OBJECTS + NUM_STATES) +
        NUM_DIRECTIONS + NUM_ACTIONS + NUM_REWARD,
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
        project="LPL-SNN-2",

        # track hyperparameters and run metadata
        config={
            "architecture": "initial",
            "dataset": "rl-memory-path",
            "settings": settings,
        }
    )

    net = Net(settings).to(device=settings.device)
    visualizer = NetworkVisualizer(net)

    env = gym.make(
        'MiniGrid-FourRooms-v0',
        render_mode='none',
        max_steps=1000000)
    env.action_space.seed(42)

    observation, info = env.reset(seed=42)
    # env.render()

    for _ in tqdm(range(INIT_TIMESTEPS), desc="preprocess"):
        # make action on environment
        action = env.action_space.sample(
            np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int8))

        # some actions aren't used
        if action > 2:
            continue

        observation, reward, terminated, truncated, info = env.step(action)

        # convert to spike encoding based on: action, reward, visibility, and direction
        # feed this into the network
        visibility = observation["image"]
        direction = observation["direction"]
        spike_encoding = convert_observation_to_spike_input(
            visibility, direction, action, reward)
        net.process_data_single_timestep(spike_encoding)
        # visualizer.update()

        if terminated or truncated:
            print("terminated: ", terminated)
            print("truncated: ", truncated)
            break

    activations_per_timestep = []
    position_x = []
    position_y = []
    rewards = []
    past_actions = []
    for _ in tqdm(range(TRAIN_TIMESTEPS), desc="collect for decoding"):
        # make action on environment
        action = env.action_space.sample(
            np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int8))

        # some actions aren't used
        if action > 2:
            continue

        observation, reward, terminated, truncated, info = env.step(action)

        # convert to spike encoding based on: action, reward, visibility, and direction
        # feed this into the network
        visibility = observation["image"]
        direction = observation["direction"]
        spike_encoding = convert_observation_to_spike_input(
            visibility, direction, action, reward)
        net.process_data_single_timestep(spike_encoding)
        # visualizer.update()

        # log information needed for decoder
        activations_per_timestep.append(net.layer_activations())
        position_x.append(env.agent_pos[0])
        position_y.append(env.agent_pos[1])
        rewards.append(reward)
        past_actions.append(action)

        # print(net.layer_activations())
        # input()

        if terminated or truncated:
            print("terminated: ", terminated)
            print("truncated: ", truncated)
            break

    activations, position_x, position_y, rewards, past_actions = smooth_for_decoding(
        activations_per_timestep, position_x, position_y, rewards, past_actions)
    decoder = train_decode_from_training_run(activations,
                                             position_x, position_y, rewards, past_actions)

    activations_inference = []
    positions_x_inference = []
    positions_y_inference = []
    rewards_inference = []
    past_actions_inference = []
    for _ in tqdm(range(INFERENCE_TIMESTEPS), desc="inference"):
        # make action on environment
        action = env.action_space.sample(
            np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int8))

        # some actions aren't used
        if action > 2:
            continue

        observation, reward, terminated, truncated, info = env.step(action)

        # convert to spike encoding based on: action, reward, visibility, and direction
        # feed this into the network
        visibility = observation["image"]
        direction = observation["direction"]
        spike_encoding = convert_observation_to_spike_input(
            visibility, direction, action, reward)
        net.process_data_single_timestep(spike_encoding)

        # log information for inference
        activations_inference.append(net.layer_activations())
        positions_x_inference.append(env.agent_pos[0])
        positions_y_inference.append(env.agent_pos[1])
        rewards_inference.append(reward)
        past_actions_inference.append(action)

        if terminated or truncated:
            print("terminated: ", terminated)
            print("truncated: ", truncated)
            break

    env.close()

    activations_inference, positions_x_inference, positions_y_inference, rewards_inference_inference, past_actions_inference \
        = smooth_for_decoding(activations_inference, positions_x_inference,
                              positions_y_inference, rewards_inference, past_actions_inference)

    with torch.no_grad():
        predicted_actions, predicted_positions_x, predicted_positions_y, predicted_rewards = decoder(
            activations_inference)
        print(f"predicted pos x shape: {predicted_positions_x.shape}")
        print(f"predicted pos y shape: {predicted_positions_y.shape}")
        print(f"predicted rewards shape: {predicted_rewards.shape}")
        print(f"predicted actions shape: {predicted_actions.shape}")
        input()

    # # Convert the predicted values to numpy arrays
    # predicted_positions = predicted_positions.numpy()
    # predicted_rewards = predicted_rewards.numpy()
    # predicted_actions = predicted_actions.numpy()

    # # Calculate the percentage of correct predictions
    # correct_positions = np.sum(np.all(predicted_positions == positions_inference, axis=1))
    # correct_rewards = np.sum(predicted_rewards == rewards_inference)
    # correct_actions = np.sum(predicted_actions == past_actions_inference)

    # total_positions = len(positions_inference)
    # total_rewards = len(rewards_inference)
    # total_actions = len(past_actions_inference)

    # position_accuracy = correct_positions / total_positions * 100
    # reward_accuracy = correct_rewards / total_rewards * 100
    # action_accuracy = correct_actions / total_actions * 100

    # Print the inference results
    print("Inference Results:")
    action_running_correct = 0
    position_running_correct = 0
    running_total = 0
    for t in range(INFERENCE_TIMESTEPS):
        print(f"Timestep: {t}")
        print(f"  Predicted Position X: {predicted_positions_x[t]}")
        print(f"  Actual Position X: {positions_x_inference[t]}")
        print(f"  Predicted Reward: {predicted_rewards[t]}")
        print(f"  Actual Reward: {rewards_inference[t]}")
        print(f"  Predicted Action: {predicted_actions[t]}")
        print(f"  Actual Action: {past_actions_inference[t]}")

        # check if argmax of predicted action is equal to actual action
        running_total += 1

        if np.argmax(predicted_actions[t]) == np.argmax(
                past_actions_inference[t]):
            action_running_correct += 1

        if np.argmax(predicted_positions_x[t]) == np.argmax(
                positions_x_inference[t]):
            position_running_correct += 1

        print(
            f"Action match prediction: {np.argmax(predicted_actions[t]) == np.argmax(past_actions_inference[t])}")
        print(
            f"Action accuracy: {action_running_correct / running_total * 100:.2f}%")

        print(
            f"Position match prediction: {np.argmax(predicted_positions_x[t]) == np.argmax(positions_x_inference[t])}")
        print(
            f"Position accuracy: {position_running_correct / running_total * 100:.2f}%")

        input()
