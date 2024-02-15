import pytest
import torch
import pandas as pd

from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import SequentialDataset
from datasets.src.zenke_2a import datagen

# Constants for testing
NUM_SAMPLES_TEST = 100  # Smaller number of samples for testing


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Pytest fixture to generate test data."""
    return datagen.generate_sequential_dataset()


def test_sequential_dataset_initialization(test_data: pd.DataFrame) -> None:
    """Test the initialization of the SequentialDataset."""
    dataframe = datagen.generate_sequential_dataset()
    dataset = SequentialDataset(dataframe)
    assert isinstance(
        dataset, SequentialDataset), "Dataset should be an instance of SequentialDataset"


def test_sequential_dataset_getitem(test_data: pd.DataFrame) -> None:
    """Test the __getitem__ method of the SequentialDataset."""
    dataframe = datagen.generate_sequential_dataset()
    dataset = SequentialDataset(dataframe)

    sample = dataset[0]
    assert torch.is_tensor(
        sample), "Data in sample should be a torch tensor"
    assert sample.shape[0] > 0, "Sample should have at least 1 row"
    assert sample.shape[1] == 2, "Sample should have 2 columns"
