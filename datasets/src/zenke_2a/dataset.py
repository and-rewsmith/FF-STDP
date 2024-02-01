import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from datasets.src.zenke_2a.constants import DATA_PATH


class SequentialDataset(Dataset):
    """Custom Dataset class for handling sequential data."""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.data_frame = data_frame

    def __len__(self) -> int:
        return len(self.data_frame) - 1  # Adjust for sequence generation

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Args:
            idx (int): Index of the data point.
        Returns:
            sample (dict): {'current': current_data, 'next': next_data}
        """
        current_data = self.data_frame.iloc[idx].to_numpy()
        next_data = self.data_frame.iloc[idx + 1].to_numpy()

        # Convert to tensors
        current_data = torch.tensor(current_data, dtype=torch.float)
        next_data = torch.tensor(next_data, dtype=torch.float)

        sample = {'current': current_data, 'next': next_data}

        return sample


if __name__ == "__main__":
    # Instantiate the dataset
    data_frame = pd.read_csv(DATA_PATH)
    sequential_dataset = SequentialDataset(data_frame=data_frame)

    # DataLoader
    data_loader = DataLoader(sequential_dataset, batch_size=1, shuffle=False)

    # For demonstration: Retrieve a batch of data
    for i, batch in enumerate(data_loader):
        print(f"Batch {i} - Sample data: {batch}")
        break  # Only showing the first batch for demonstration purposes
