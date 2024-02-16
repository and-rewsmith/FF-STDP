import pandas as pd
from typing import List, Optional, Self

from torch import nn
from torch.utils.data import DataLoader
import torch
import snntorch as snn

from datasets.src.zenke_2a.constants import TEST_DATA_PATH, TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import SequentialDataset
from model.src.util import TemporalFilter


class Settings:
    def __init__(self,
                 layer_sizes: list[int],
                 beta: float,
                 learn_beta: bool,
                 num_steps: int,
                 data_size: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int) -> None:
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.learn_beta = learn_beta
        self.num_steps = num_steps
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs


class LayerSettings:
    def __init__(self, prev_size: int, size: int, next_size: int, beta: float, learn_beta: bool,
                 batch_size: int, learning_rate: float) -> None:
        self.prev_size = prev_size
        self.size = size
        self.next_size = next_size
        self.beta = beta
        self.learn_beta = learn_beta
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class Layer(nn.Module):
    def __init__(self, layer_settings: LayerSettings) -> None:
        super().__init__()

        # weights from prev layer to this layer
        self.forward_weights = nn.Linear(
            layer_settings.prev_size, layer_settings.size)
        self.forward_lif = snn.Leaky(
            beta=layer_settings.beta, learn_beta=layer_settings.learn_beta)

        # TODO: cap size
        self.spk_rec: List[torch.Tensor] = []
        self.mem_rec: List[torch.Tensor] = []

        self.prev_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None

        self.mem = self.forward_lif.init_leaky()

        # TODO PREMERGE: we need to pick better tau values
        self.alpha_filter = TemporalFilter(tau_rise=1, tau_fall=1)
        self.epsilon_filter = TemporalFilter(tau_rise=1, tau_fall=1)
        self.spike_moving_average = TemporalFilter(tau_rise=1, tau_fall=1)
        self.variance_moving_average = TemporalFilter(tau_rise=1, tau_fall=1)

        # self.optimizer = torch.optim.Adam(
        #     self.forward_weights.parameters(), lr=layer_settings.learning_rate)

    def set_next_layer(self, next_layer: Self) -> None:
        self.next_layer = next_layer

    def set_prev_layer(self, prev_layer: Self) -> None:
        self.prev_layer = prev_layer

    def forward(self, data: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if data is None:
            assert self.prev_layer is not None
            current = self.forward_weights(
                self.prev_layer.spk_rec[-1].detach())
        else:
            data = data.detach()
            current = self.forward_weights(data)

        spk, mem = self.forward_lif(current, self.mem)
        self.spk_rec.append(spk)
        self.mem_rec.append(mem)

        return spk, mem

    def train_forward(self, data: Optional[torch.Tensor] = None) -> None:
        # self.optimizer.zero_grad()

        spk, _mem = self.forward(data)
        # loss = torch.sum(spk)
        # loss.backward()

        # self.optimizer.step()


class Net(nn.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()

        self.settings = settings

        # make settings for each layer
        network_layer_settings = []
        for i, size in enumerate(settings.layer_sizes):
            prev_size = settings.data_size if i == 0 else settings.layer_sizes[i-1]
            next_size = settings.layer_sizes[i+1] if i < len(
                settings.layer_sizes) - 1 else 0
            layer_settings = LayerSettings(
                prev_size, size, next_size, settings.beta, settings.learn_beta,
                settings.batch_size, settings.learning_rate)
            network_layer_settings.append(layer_settings)

        # make layers
        self.layers = nn.ModuleList()
        for i, layer_spec in enumerate(network_layer_settings):
            layer = Layer(layer_spec)
            self.layers.append(layer)

        # connect layers
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.set_prev_layer(self.layers[i-1])
            if i < len(network_layer_settings) - 1:
                layer.set_next_layer(self.layers[i+1])

    # TODO: handle test data
    def process_data_online(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        for epoch in range(self.settings.epochs):
            for i, batch in enumerate(train_loader):
                batch = batch.permute(1, 0, 2)
                batch = batch[:, :, :1]

                print(
                    f"Epoch {epoch} - Batch {i} - Sample data: {batch.shape}")

                for _timestep in range(batch.shape[0]):
                    for i, layer in enumerate(self.layers):
                        if i == 0:
                            layer.train_forward(batch)
                        else:
                            layer.train_forward(None)


if __name__ == "__main__":

    settings = Settings(
        layer_sizes=[784, 300, 10],
        beta=0.9,
        learn_beta=False,
        num_steps=10,
        data_size=1,
        batch_size=10,
        learning_rate=0.01,
        epochs=10
    )

    train_dataframe = pd.read_csv(TRAIN_DATA_PATH)
    train_sequential_dataset = SequentialDataset(train_dataframe)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=10, shuffle=False)

    test_dataframe = pd.read_csv(TEST_DATA_PATH)
    test_sequential_dataset = SequentialDataset(test_dataframe)
    test_data_loader = DataLoader(
        test_sequential_dataset, batch_size=10, shuffle=False)

    net = Net(settings)
    net.process_data_online(train_data_loader, test_data_loader)
