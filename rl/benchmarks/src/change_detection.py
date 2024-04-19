import gymnasium as gym
import random
from torch import nn

from rl.benchmarks.src.change_detection_framework import ChangeDetectionBasic

"""
TODO: 
implement actor critic
assess lpl decodability on trained policy (train policy helps decoding the reward)
implement the bridge from state to the LPL network, and from LPL network to actor / critic
batch processing
if that doesn't work, implement some sort of search

verify metaplasticity

"""

ACTION_DIM = 1
ACTOR_LR = 1e-6 * 5
CRITIC_LR = 1e-6 * 5

HIDDEN_LAYER_SIZE = 2048
LAST_LAYER_SIZE = 32


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


def main():
    # Create the ChangeDetectionBasic environment
    env = ChangeDetectionBasic()

    # Set the number of episodes to run
    num_episodes = 5

    # Run the episodes
    for episode in range(num_episodes):
        # Reset the environment for a new episode
        observation = env.reset()
        done = False
        total_reward = 0

        # Run the episode until done
        while not done:
            # Choose an action (0 for no lick, 1 for lick)
            action = random.choice(env.action_space)  # Randomly select an action from the action_space list

            # Take a step in the environment
            observation, reward, done, info = env.step(action)

            # Update the total reward for the episode
            total_reward += reward

            # Print the current step information
            print(
                f"Episode: {episode+1}, Time: {info['time']:.2f}, Action: {action}, Observation: {observation}, Reward: {reward}")

        # Print the episode summary
        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Completed Trials: {info['completed_trials']}\n")

    # Close the environment
    env.close()


if __name__ == '__main__':
    main()
