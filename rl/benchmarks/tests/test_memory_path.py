import pytest
import torch
import numpy as np

from rl.benchmarks.src.memory_path import convert_observation_to_spike_input

NUM_OBJECTS = 11
NUM_COLORS = 6
NUM_STATES = 3
NUM_ACTIONS = 3
NUM_DIRECTIONS = 4
# reward is a float that is one hot encoded
NUM_REWARD = 1


def test_convert_observation_to_spike_input():
    """
    Test case 1: Basic input
    Tests the function with a basic input. It checks that the returned result is a torch.Tensor,
    has the expected shape, and the sum of the vision binary array and the multi-hot encoded
    direction and action values match the expected values.
    """
    vision = np.array([
        [[1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 3, 0],
            [5, 4, 0], [6, 5, 0], [7, 0, 1]],
        [[8, 1, 1], [9, 2, 1], [1, 0, 0], [2, 1, 0],
            [3, 2, 0], [4, 3, 0], [5, 4, 0]],
        [[6, 5, 0], [7, 0, 1], [8, 1, 1], [9, 2, 1],
            [1, 0, 0], [2, 1, 0], [3, 2, 0]],
        [[4, 3, 0], [5, 4, 0], [6, 5, 0], [7, 0, 1],
            [8, 1, 1], [9, 2, 1], [1, 0, 0]],
        [[2, 1, 0], [3, 2, 0], [4, 3, 0], [5, 4, 0],
            [6, 5, 0], [7, 0, 1], [8, 1, 1]],
        [[9, 2, 1], [1, 0, 0], [2, 1, 0], [3, 2, 0],
            [4, 3, 0], [5, 4, 0], [6, 5, 0]],
        [[7, 0, 1], [8, 1, 1], [9, 2, 1], [1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 3, 0]]
    ])
    direction = 1
    action = 2
    reward = 1
    expected_shape = (1, 7 *
                      7 *
                      (NUM_OBJECTS +
                       NUM_COLORS +
                       NUM_STATES) +
                      NUM_DIRECTIONS +
                      NUM_ACTIONS +
                      NUM_REWARD)

    result = convert_observation_to_spike_input(
        vision, direction, action, reward)

    assert isinstance(result, torch.Tensor)
    assert result.shape == expected_shape

    assert torch.sum(result[:, :7 * 7 * (NUM_OBJECTS +
                     NUM_COLORS + NUM_STATES)]).item() == vision.size
    assert torch.sum(
        result[:, 7 * 7 * (NUM_OBJECTS + NUM_COLORS + NUM_STATES):]).item() == 3


def test_convert_observation_to_spike_input_no_reward():
    """
    Test case 1: Basic input
    Tests the function with a basic input. It checks that the returned result is a torch.Tensor,
    has the expected shape, and the sum of the vision binary array and the multi-hot encoded
    direction and action values match the expected values.
    """
    vision = np.array([
        [[1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 3, 0],
            [5, 4, 0], [6, 5, 0], [7, 0, 1]],
        [[8, 1, 1], [9, 2, 1], [1, 0, 0], [2, 1, 0],
            [3, 2, 0], [4, 3, 0], [5, 4, 0]],
        [[6, 5, 0], [7, 0, 1], [8, 1, 1], [9, 2, 1],
            [1, 0, 0], [2, 1, 0], [3, 2, 0]],
        [[4, 3, 0], [5, 4, 0], [6, 5, 0], [7, 0, 1],
            [8, 1, 1], [9, 2, 1], [1, 0, 0]],
        [[2, 1, 0], [3, 2, 0], [4, 3, 0], [5, 4, 0],
            [6, 5, 0], [7, 0, 1], [8, 1, 1]],
        [[9, 2, 1], [1, 0, 0], [2, 1, 0], [3, 2, 0],
            [4, 3, 0], [5, 4, 0], [6, 5, 0]],
        [[7, 0, 1], [8, 1, 1], [9, 2, 1], [1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 3, 0]]
    ])
    direction = 1
    action = 2
    reward = 0
    expected_shape = (1, 7 *
                      7 *
                      (NUM_OBJECTS +
                       NUM_COLORS +
                       NUM_STATES) +
                      NUM_DIRECTIONS +
                      NUM_ACTIONS +
                      NUM_REWARD)

    result = convert_observation_to_spike_input(
        vision, direction, action, reward)

    assert isinstance(result, torch.Tensor)
    assert result.shape == expected_shape

    assert torch.sum(result[:, :7 * 7 * (NUM_OBJECTS +
                     NUM_COLORS + NUM_STATES)]).item() == vision.size
    assert torch.sum(
        result[:, 7 * 7 * (NUM_OBJECTS + NUM_COLORS + NUM_STATES):]).item() == 2


def test_convert_observation_to_spike_input_empty():
    """
    Test case 2: Empty input
    Tests the function with an empty vision input (all zeros). It checks that the returned
    result is a torch.Tensor, has the expected shape, and the sum of the entire tensor is 2
    (corresponding to the one-hot encoded direction and action).
    """
    vision = np.zeros((7, 7, 3), dtype=int)
    direction = 0
    action = 0
    reward = 1
    expected_shape = (1, 7 *
                      7 *
                      (NUM_OBJECTS +
                       NUM_COLORS +
                       NUM_STATES) +
                      NUM_DIRECTIONS +
                      NUM_ACTIONS +
                      NUM_REWARD)

    result = convert_observation_to_spike_input(
        vision, direction, action, reward)

    assert isinstance(result, torch.Tensor)
    assert result.shape == expected_shape
    assert torch.sum(result).item() == 7 * 7 * 3 + 3


def test_convert_observation_to_spike_input_invalid_direction():
    """
    Test case 3: Invalid direction input
    Tests the function with an invalid direction input. It expects an IndexError to be raised.
    """
    vision = np.zeros((7, 7, 3), dtype=int)
    direction = 4  # Invalid direction
    action = 0
    reward = 0

    with pytest.raises(IndexError):
        convert_observation_to_spike_input(vision, direction, action, reward)


def test_convert_observation_to_spike_input_invalid_action():
    """
    Test case 4: Invalid action input
    Tests the function with an invalid action input. It expects an IndexError to be raised.
    """
    vision = np.zeros((7, 7, 3), dtype=int)
    direction = 0
    action = 3  # Invalid action
    reward = 0

    with pytest.raises(IndexError):
        convert_observation_to_spike_input(vision, direction, action, reward)
