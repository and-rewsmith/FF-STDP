import torch


class Settings:
    def __init__(self,
                 layer_sizes: list[int],
                 data_size: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 encode_spike_trains: bool,
                 device: torch.device) -> None:
        self.layer_sizes = layer_sizes
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.encode_spike_trains = encode_spike_trains
        self.device = device


class LayerSettings:
    def __init__(self, layer_id: int, prev_size: int, size: int, next_size: int,
                 batch_size: int, learning_rate: float, data_size: int, device: torch.device) -> None:
        self.layer_id = layer_id
        self.prev_size = prev_size
        self.size = size
        self.next_size = next_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_size = data_size
        self.device = device
