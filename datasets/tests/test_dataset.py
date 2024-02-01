import pytest
import torch
import pandas as pd
from datasets.src.zenke_2a.dataset import SequentialDataset
from datasets.src.zenke_2a import datagen

# Constants for testing
NUM_SAMPLES_TEST = 100  # Smaller number of samples for testing


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Pytest fixture to generate test data."""
    return datagen.generate_sequential_dataset(num_samples=NUM_SAMPLES_TEST)


def test_sequential_dataset_initialization(test_data: pd.DataFrame) -> None:
    """Test the initialization of the SequentialDataset."""
    dataset = SequentialDataset(data_frame=test_data)
    assert isinstance(
        dataset, SequentialDataset), "Dataset should be an instance of SequentialDataset"


def test_sequential_dataset_length(test_data: pd.DataFrame) -> None:
    """Test the __len__ method of the SequentialDataset."""
    dataset = SequentialDataset(data_frame=test_data)
    assert len(dataset) == NUM_SAMPLES_TEST - \
        1, f"Length of dataset should be {NUM_SAMPLES_TEST - 1}"


def test_sequential_dataset_getitem(test_data: pd.DataFrame) -> None:
    """Test the __getitem__ method of the SequentialDataset."""
    dataset = SequentialDataset(data_frame=test_data)
    sample = dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "current" in sample and "next" in sample, "Sample should have 'current' and 'next' keys"
    assert torch.is_tensor(sample["current"]) and torch.is_tensor(
        sample["next"]), "Data in sample should be torch tensors"
    assert sample["current"].shape == sample["next"].shape, "Shapes of 'current' and 'next' tensors should be the same"
