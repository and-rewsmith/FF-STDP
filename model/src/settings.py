class Settings:
    def __init__(self,
                 layer_sizes: list[int],
                 num_steps: int,
                 data_size: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 encode_spike_trains: bool) -> None:
        self.layer_sizes = layer_sizes
        self.num_steps = num_steps
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.encode_spike_trains = encode_spike_trains


class LayerSettings:
    def __init__(self, prev_size: int, size: int, next_size: int,
                 batch_size: int, learning_rate: float, data_size: int) -> None:
        self.prev_size = prev_size
        self.size = size
        self.next_size = next_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_size = data_size
