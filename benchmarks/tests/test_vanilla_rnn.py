import torch
import torch.nn as nn

from model.src.settings import LayerSettings, Settings
from model.src.util import MovingAverageLIF


class VanillaSpikingRNN(nn.Module):
    def __init__(self, settings: Settings):
        super().__init__()

        # make settings for each layer
        network_layer_settings = []
        for i, size in enumerate(settings.layer_sizes):
            prev_size = settings.data_size if i == 0 else settings.layer_sizes[i - 1]
            next_size = settings.layer_sizes[i + 1] if i < len(settings.layer_sizes) - 1 else 0
            layer_id = i
            layer_settings = LayerSettings(layer_id,
                                           prev_size,
                                           size,
                                           next_size,
                                           settings.batch_size,
                                           settings.learning_rate,
                                           settings.data_size,
                                           settings.dt,
                                           settings.percentage_inhibitory,
                                           settings.exc_to_inhib_conn_c,
                                           settings.exc_to_inhib_conn_sigma_squared,
                                           settings.layer_sparsity,
                                           settings.decay_beta,
                                           settings.threshold_scale,
                                           settings.threshold_decay,
                                           settings.tau_mean,
                                           settings.tau_var,
                                           settings.tau_stdp,
                                           settings.tau_rise_alpha,
                                           settings.tau_fall_alpha,
                                           settings.tau_rise_epsilon,
                                           settings.tau_fall_epsilon,
                                           settings.device)
            network_layer_settings.append(layer_settings)

        # Initialize layers directly in a list since no training parameters are used
        self.layers = [MovingAverageLIF(layer_settings) for layer_settings in network_layer_settings]

        # Fixed connections as regular tensors since they do not need to be parameters
        sizes_with_data_size = [settings.data_size] + settings.layer_sizes
        self.connections = []
        for size, next_size in zip(sizes_with_data_size[:-1], sizes_with_data_size[1:]):
            self.connections.append(torch.rand(size, next_size, device=settings.device))

    def process_data_single_timestep(self, data):
        current_input = data.unsqueeze(0) @ self.connections[0]
        for i, layer in enumerate(self.layers):
            spikes = layer.forward(current_input)

            if i != len(self.layers) - 1:
                current_input = torch.matmul(spikes, self.connections[i+1])

        self.layers[-1].forward(current_input)

    def layer_activations(self):
        # Collect and return the latest spike recordings from each layer
        return [layer.spike_moving_average.spike_rec[-1] for layer in self.layers]


def test_vanilla_spiking_rnn():
    settings = Settings(
        layer_sizes=[2, 4, 6],
        data_size=50,
        batch_size=1,
        learning_rate=0.01,
        epochs=10,
        encode_spike_trains=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model = VanillaSpikingRNN(settings)

    input_data = torch.randn(settings.data_size, device=settings.device)
    for i in range(10):
        model.process_data_single_timestep(input_data)
    spikes = model.layer_activations()
    print("Latest spikes from each layer:", spikes)


if __name__ == "__main__":
    test_vanilla_spiking_rnn()
