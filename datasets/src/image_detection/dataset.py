from enum import Enum
from typing import List, Union
import random

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self,
                 num_timesteps_each_image: int,
                 num_timesteps_flash: int,
                 num_switches: int,
                 switch_probability: float,
                 device: str,
                 max_samples: int = 1024 * 10) -> None:
        self.num_classes = 10
        self.num_timesteps_each_image = num_timesteps_each_image
        self.num_timesteps_flash = num_timesteps_flash
        self.num_switches = num_switches
        self.images = [num for num in range(0, self.num_classes)]
        self.switch_probability = switch_probability
        self.len = max_samples
        self.device = device

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.len

    def __getitem__(self,
                    idx: Union[int,
                               List[int],
                               torch.Tensor]) -> torch.Tensor:
        initial_image_index = random.randint(0, len(self.images)-1)
        switch_image_index = -1
        while switch_image_index == -1 or initial_image_index == switch_image_index:
            switch_image_index = random.randint(0, len(self.images)-1)

        out = []
        out_labels = []
        switch = False
        for i in range(self.num_switches):
            if switch:
                out.extend([switch_image_index] * self.num_timesteps_each_image)
                out_labels.append(switch_image_index)
            else:
                out.extend([initial_image_index] * self.num_timesteps_each_image)
                out_labels.append(initial_image_index)

            if i != self.num_switches - 1:
                out.extend([switch_image_index] * self.num_timesteps_flash)

            if not switch:
                switch = random.random() < self.switch_probability

        # tensor is of shape (batch_size, num_timesteps), but we want to convert to (batch_size, num_timesteps, 1)
        out = torch.tensor(out, dtype=torch.float, device=self.device)
        return out.unsqueeze(-1), torch.tensor(out_labels, dtype=torch.long, device=self.device)
