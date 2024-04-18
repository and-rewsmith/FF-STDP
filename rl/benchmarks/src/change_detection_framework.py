import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np


class ChangeDetectionBasic(gym.Env):
    metadata = {'render.modes': ['human']}  # is this what I want?

    def __init__(self, duration=3600, timestep=0.001, timeout=0.3, flash_duration=0.25,
                 # trial logic should make this effectively 3.75 (total time from trial state = 4.5)
                 blank_duration=0.5, grace_period=3.5,
                 response_window=(0.15, 0.75), number_of_stimuli=8, max_attempts=5,
                 flash_change_window=(5, 11),  # inclusive of lower bound, exclusive of upper
                 change_probability_rate=0.3):

        self.duration = duration
        self.timestep = timestep
        self.timeout = timeout
        self.flash_duration = flash_duration
        self.blank_duration = blank_duration
        self.flash_blank_duration = self.flash_duration + self.blank_duration
        self.grace_period = grace_period
        self.response_window = response_window
        self.ignore_duration = self.response_window[0]
        self.response_duration = self.response_window[1]
        self.number_of_stimuli = number_of_stimuli
        self.max_attempts = max_attempts

        # use gym.spaces for these?
        self.stimuli = list(range(self.number_of_stimuli))
        self.observation_space = ['blank'] + self.stimuli
        self.action_space = [0, 1]  # [no_lick, lick]

        window_length = flash_change_window[1] - flash_change_window[0]
        prob = change_probability_rate*((1.0-change_probability_rate)**np.arange(0, window_length))
        prob = prob/np.sum(prob)
        self.flash_probability = np.zeros(flash_change_window[1])
        self.flash_probability[flash_change_window[0]:] = prob
        self.cum_flash_probability = np.cumsum(self.flash_probability)

        self.reset()

    def step(self, action):
        reward = 0
        observation = None
        done = False

        # print("stepping.  v = {}".format(self.v))
        self.time += self.timestep
        dt_flash = self.time - self.last_flash_start

        if self.time >= self.duration:
            done = True

        # check to see if we need to start a new flash and update the stimulus if the next flash is a "change" flash
        if dt_flash >= self.flash_blank_duration:
            # self.completed_flashes += 1
            self.last_flash_start += self.flash_blank_duration
            dt_flash = self.time - self.last_flash_start  # reset dt_flash
            if self.completed_flashes >= self.change_on_flash and not self.changed:
                self.last_change_start = self.last_flash_start
                self.current_stimulus = self.next_stimulus
                self.changed = True

        #  update stimulus if necessary
        if dt_flash < self.flash_duration:
            observation = self.current_stimulus
        if dt_flash >= self.flash_duration and dt_flash < self.flash_blank_duration:
            observation = 'blank'

        # decide upon state for reward logic
        if self.last_change_start is not None:
            dt_change = self.time - self.last_change_start
            if dt_change < self.ignore_duration:
                self.reward_state = 'ignore'
            elif dt_change < self.response_duration:
                if self.trial_type == 'go':
                    self.reward_state = 'reward'
            else:
                self.reward_state = 'grace'
        elif self.reward_state != 'timeout':
            self.reward_state = 'abort'

        # evaluate consequences of action (1==licking, 0==no licking)
        if self.reward_state == 'timeout' and action == 1:
            # set new trial start at next onset flash that is longer than self.timeout away
            # self.last_trial_start = self.last_flash_start + self.flash_blank_duration # start new trial on next flash onset
            if self.last_trial_start - self.time < self.timeout:
                self.last_trial_start += self.flash_blank_duration  # start new trial on next flash onset

        if self.reward_state == 'abort':
            if action == 1:
                self.reward_state = 'timeout'
                self.number_of_aborts += 1

                # set new trial start at next onset flash that is longer than self.timeout away
                self.last_trial_start = self.last_flash_start + self.flash_blank_duration  # start new trial on next flash onset
                if self.last_trial_start - self.time < self.timeout:
                    self.last_trial_start += self.flash_blank_duration  # start new trial on next flash onset

        if self.reward_state == 'reward':
            if action == 1 and self.rewarded == False:
                reward = 1
                self.rewarded = True

        info = {'rewarded':  self.rewarded,
                'reward_state':  self.reward_state,
                'number_of_aborts':  self.number_of_aborts,
                'trial_type':  self.trial_type,
                'completed_flashes':  self.completed_flashes,
                'completed_trials':  self.completed_trials,
                'time':  self.time,
                'last_flash_start':  self.last_flash_start,
                'last_trial_start':  self.last_trial_start,
                'last_change_start':  self.last_change_start,
                'change_on_flash':  self.change_on_flash,
                'initial_stimulus':  self.initial_stimulus,
                'final_stimulus':  self.next_stimulus,
                'current_stimulus':  self.current_stimulus,
                'trial_complete':  False,
                'changed':  self.changed,
                'action':  action}

        dt_trial = self.time - self.last_trial_start
        # start a new trial when this one is through
        if self.last_change_start is not None:
            if self.time > self.last_change_start + self.response_duration + self.grace_period:
                if dt_trial > 0:
                    self.last_trial_start = self.last_flash_start + self.flash_blank_duration  # start new trial on next flash onset
                    dt_trial = self.time - self.last_trial_start
                # dt_trial (below) will be negative until we start the new trial so this if block will be skipped

        if dt_flash + self.timestep >= self.flash_blank_duration:
            self.completed_flashes += 1
            info['completed_flashes'] = self.completed_flashes

        # start new trial when it is time
        if dt_trial >= -self.timestep and dt_trial <= 0.0:
            # do I need an end trial function?
            if self.reward_state == 'timeout' and self.number_of_aborts < self.max_attempts:
                self.start_same_trial()
            else:
                self.start_new_trial()
                info['trial_complete'] = True
                self.completed_trials += 1
                info['completed_trials'] = self.completed_trials

        return observation, reward, done, info

    def reset(self):
        self.completed_trials = 0
        self.time = 0
        self.start_new_trial()
        self.last_flash_start = 0
        self.last_trial_start = 0

    def start_new_trial(self):
        self.change_on_flash = self.get_next_change_flash()
        # self.next_change_time = self.get_next_change_time()
        self.initial_stimulus = self.get_next_stimulus()
        self.next_stimulus = self.get_next_stimulus()
        if self.next_stimulus != self.initial_stimulus:
            self.trial_type = 'go'
        else:
            self.trial_type = 'catch'
        self.current_stimulus = self.initial_stimulus
        # self.trial_complete = False

        self.number_of_aborts = 0
        self.start_same_trial()

    def start_same_trial(self):
        self.completed_flashes = 0
        self.reward_state = 'abort'

        self.last_change_start = None
        self.rewarded = False
        self.changed = False

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_next_change_flash(self):
        """return next change in number of completed flashes"""  # time in seconds from the beginning of the trial"""
        p = np.random.random()
        return np.where(p < self.cum_flash_probability)[0][0]

    def get_next_stimulus(self):  # , previous=None):
        # if previous is not None:
        #     return np.random.choice(self.stimuli[0:previous]+self.stimuli[previous+1:])
        # else:
        return np.random.choice(self.stimuli)
