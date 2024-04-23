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
    # TODOPRE: adjust durations
    def __init__(self, batch_size=1, flash_duration=25, blank_duration=25, ignore_ratio=0.2, number_of_possible_stimuli=8, change_probability=0.25, max_flash_alternations=6, device="cpu"):
        self.batch_size = batch_size
        self.flash_duration = flash_duration
        self.blank_duration = blank_duration
        self.ignore_ratio = ignore_ratio
        self.number_of_stimuli = number_of_possible_stimuli
        self.change_probability = change_probability
        self.max_flash_alternations = max_flash_alternations
        self.device = device

        self.observation_space = list(range(self.number_of_stimuli))
        self.action_space = [0, 1]  # [no_lick, lick]

        self.reset()

    def step(self, action: torch.Tensor):
        # Ensure the action tensor has the correct shape
        assert action.shape == (
            self.batch_size,), f"Expected action shape: ({self.batch_size},), but got: {action.shape}"

        # Update time
        self.time += 1

        # Check if it's time to switch between stimulus and blank
        if self.blank_showing and np.isclose(self.time % (self.flash_duration + self.blank_duration), 0):
            print("------changing stimulus")
            self.blank_showing = False
            # Change to the next stimulus with a certain probability
            valid_change_stimulus = torch.rand(self.batch_size).to(device=self.device) < self.change_probability
            valid_change_stimulus.logical_and_((self.current_stimulus == self.next_stimulus).logical_not_())
            valid_change_stimulus.logical_and_((self.reward_state == STATE_TIMEOUT).logical_not_())
            self.current_stimulus = torch.where(valid_change_stimulus, self.next_stimulus, self.current_stimulus)
            self.reward_state = torch.where(valid_change_stimulus, torch.tensor(
                STATE_IGNORE), self.reward_state).to(device=self.device)  # Enter ignore state if stimulus changed
        elif not self.blank_showing and np.isclose(self.time % (self.flash_duration + self.blank_duration), self.flash_duration):
            print("------showing blank")
            self.blank_showing = True
            self.flash_alternations += 1

        # Transition from ignore state to reward state
        ignore_time = self.flash_duration * self.ignore_ratio
        # print(f"ignore_time: {ignore_time}")
        # print(f"flash_duration: {self.flash_duration}")
        # print(f"blank_duration: {self.blank_duration}")
        # print(
        #     f"self.time % (self.flash_duration + self.blank_duration): {self.time % (self.flash_duration + self.blank_duration)}")
        # print(f"reward state: {self.reward_state}")
        self.reward_state = torch.where((self.reward_state == STATE_IGNORE).to(device=self.device) & (self.time % (
            self.flash_duration + self.blank_duration) >= ignore_time), torch.tensor(STATE_REWARD).to(device=self.device), self.reward_state)

        # Determine the current observation
        blank_showing_tensor = torch.tensor([self.blank_showing] * self.batch_size,
                                            dtype=torch.bool).to(device=self.device)
        observation = torch.where(blank_showing_tensor, torch.tensor(-1).to(device=self.device), self.current_stimulus)

        # Update reward state based on the agent's action
        lick_action = action == 1
        correct_lick = lick_action & torch.eq(self.reward_state, STATE_REWARD)
        incorrect_lick = lick_action & (torch.eq(self.reward_state, STATE_REWARD) |
                                        torch.eq(self.reward_state, STATE_IGNORE)).logical_not_()
        self.reward_state = torch.where(incorrect_lick, torch.tensor(
            STATE_TIMEOUT).to(device=self.device), self.reward_state)  # Enter timeout state if incorrect lick
        self.reward_state = torch.where(correct_lick, torch.tensor(
            STATE_SUCCESS).to(device=self.device), self.reward_state)  # Enter success state if correct lick

        # Calculate reward
        reward = torch.where(correct_lick, torch.tensor(1.0).to(
            device=self.device), torch.tensor(0.0).to(device=self.device))

        # Check if the episode is done
        done = (self.flash_alternations >= self.max_flash_alternations)

        # Prepare info dict
        info = {
            'current_stimulus': self.current_stimulus,
            'reward_state': self.reward_state,
            'blank_showing': self.blank_showing,
        }

        return observation, reward, done, info

    def reset(self):
        self.time = 0
        self.flash_alternations = 0

        initial_stimulus_random_indices = torch.randint(
            0, len(self.observation_space), (self.batch_size,)).to(device=self.device)
        self.initial_stimulus = torch.Tensor([self.observation_space[i]
                                             for i in initial_stimulus_random_indices]).to(device=self.device)
        self.current_stimulus = self.initial_stimulus

        next_stimulus_offsets = torch.randint(1, len(self.observation_space), (self.batch_size,)).to(device=self.device)
        self.next_stimulus = (self.current_stimulus + next_stimulus_offsets) % len(self.observation_space)

        self.reward_state = torch.zeros(self.batch_size).to(device=self.device)

        self.blank_showing = False

    # needed for the gym interface
    def render(self, mode='human'):
        pass

    # needed for the gym interface
    def close(self):
        pass
