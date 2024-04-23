import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import torch

# State type constants
STATE_ABORT = 0
STATE_TIMEOUT = 1
STATE_IGNORE = 2
STATE_REWARD = 3
STATE_SUCCESS = 4


class ChangeDetectionBasic(gym.Env):

    # all time in seconds
    def __init__(self, batch_size=1, flash_duration=0.25, blank_duration=0.5, ignore_ratio=0.2, number_of_stimuli=8, change_probability=0.3):
        self.batch_size = batch_size
        self.flash_duration = flash_duration
        self.blank_duration = blank_duration
        self.ignore_ratio = ignore_ratio
        self.number_of_stimuli = number_of_stimuli
        self.change_probability = change_probability

        self.observation_space = list(range(self.number_of_stimuli))
        self.action_space = [0, 1]  # [no_lick, lick]

        self.reset()

    # step is assumed to be every 1ms
    def step(self, action: torch.Tensor):
        # Ensure the action tensor has the correct shape
        assert action.shape == (
            self.batch_size,), f"Expected action shape: ({self.batch_size},), but got: {action.shape}"

        # Update time
        self.time += 1

        # Check if it's time to switch between stimulus and blank
        if self.blank_showing and self.time % (self.flash_duration + self.blank_duration) == self.blank_duration:
            self.blank_showing = False
            # Change to the next stimulus with a certain probability
            change_stimulus = torch.rand(self.batch_size) < self.change_probability
            self.current_stimulus = torch.where(change_stimulus, self.next_stimulus, self.current_stimulus)
            self.reward_state = torch.where(change_stimulus, torch.tensor(
                STATE_IGNORE), self.reward_state)  # Enter ignore state if stimulus changed
        elif not self.blank_showing and self.time % (self.flash_duration + self.blank_duration) == self.flash_duration:
            self.blank_showing = True

        # Transition from ignore state to reward state
        ignore_time = self.flash_duration * self.ignore_ratio
        self.reward_state = torch.where((self.reward_state == STATE_IGNORE) & (self.time % (
            self.flash_duration + self.blank_duration) == ignore_time), torch.tensor(STATE_REWARD), self.reward_state)

        # Determine the current observation
        observation = torch.where(self.blank_showing, torch.tensor(-1), self.current_stimulus)

        # Update reward state based on the agent's action
        lick_action = action == 1
        correct_lick = lick_action & (self.reward_state == STATE_REWARD)
        incorrect_lick = lick_action & (self.reward_state != STATE_REWARD)
        self.reward_state = torch.where(incorrect_lick, torch.tensor(
            STATE_TIMEOUT), self.reward_state)  # Enter timeout state if incorrect lick
        self.reward_state = torch.where(correct_lick, torch.tensor(
            STATE_SUCCESS), self.reward_state)  # Enter success state if correct lick

        # Calculate reward
        reward = torch.where(correct_lick, torch.tensor(1.0), torch.tensor(0.0))

        # Check if the episode is done
        # TODO: mark the ones that have reached the reward or timeout state as
        # done, then stop processing actions (or think about best interface for
        # this)
        done = torch.zeros(self.batch_size, dtype=torch.bool)  # You can define your own termination criteria here

        # Prepare info dict
        info = {
            'current_stimulus': self.current_stimulus,
            'reward_state': self.reward_state,
            'blank_showing': self.blank_showing,
        }

        return observation, reward, done, info

    def reset(self):
        self.time = torch.zeros_like(self.batch_size)

        initial_stimulus_random_indices = torch.randoint(0, len(self.observation_space), (self.batch_size,))
        self.initial_stimulus = torch.Tensor([self.observation_space[i] for i in initial_stimulus_random_indices])
        self.current_stimulus = self.initial_stimulus

        next_stimulus_offsets = torch.randint(1, len(self.observation_space), (self.batch_size,))
        self.next_stimulus = (self.current_stimulus + next_stimulus_offsets) % len(self.observation_space)

        self.reward_state = torch.zeros_like(self.batch_size)

        self.blank_showing = False

    # needed for the gym interface
    def render(self, mode='human'):
        pass

    # needed for the gym interface
    def close(self):
        pass
