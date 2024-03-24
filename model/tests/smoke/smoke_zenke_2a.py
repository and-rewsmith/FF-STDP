import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src.constants import DT, EXC_TO_INHIB_CONN_C, EXC_TO_INHIB_CONN_SIGMA_SQUARED, LAYER_SPARSITY, LEARNING_RATE, PERCENTAGE_INHIBITORY
from model.src.layer import Layer
from model.src.logging_util import set_logging
from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer

THIS_TEST_NUM_SAMPLES = 10
THIS_TEST_NUM_DATAPOINTS = 7000


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1138)
    torch.set_printoptions(precision=10, sci_mode=False)

    set_logging()

    settings = Settings(
        layer_sizes=[2, 5],
        data_size=2,
        batch_size=THIS_TEST_NUM_SAMPLES,
        learning_rate=0.01,
        epochs=10,
        encode_spike_trains=ENCODE_SPIKE_TRAINS,
        dt=.01,
        percentage_inhibitory=50,
        exc_to_inhib_conn_c=0.25,
        exc_to_inhib_conn_sigma_squared=60,
        layer_sparsity=0.9,
        decay_beta=0.85,
        tau_mean=1200,
        tau_var=0.02,
        tau_stdp=0.1,
        tau_rise_alpha=0.005,
        tau_fall_alpha=.05,
        tau_rise_epsilon=0.002,
        tau_fall_epsilon=0.02,
        device=torch.device("cpu")
    )

    wandb.init(
        # set the wandb project where this run will be logged
        project="LPL-SNN-2",

        # track hyperparameters and run metadata
        config={
            "architecture": "initial",
            "dataset": "point-cloud",
            "settings": settings,
        }
    )

    try:
        train_dataframe = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        train_dataframe = None
    train_sequential_dataset = SequentialDataset(DatasetType.TRAIN,
                                                 train_dataframe, num_timesteps=THIS_TEST_NUM_DATAPOINTS,
                                                 planned_batch_size=settings.batch_size)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=settings.batch_size, shuffle=False)

    net = Net(settings).to(settings.device)

    # NOTE: uncomment if you want to visualize the network
    # visualizer = NetworkVisualizer(net)
    # visualizer.update()

    # Figure out what the excitatory mask is, find that neuron, then find the
    # weights for that neuron.
    layer: Layer = net.layers[0]
    weights = layer.forward_weights.weight()
    mask = layer.excitatory_mask_vec
    mask_expanded = mask.unsqueeze(
        1).expand(-1, layer.layer_settings.data_size)
    # This will eliminate all weights that are not connecting to an excitatory
    # neuron, then collapse the dim down to 1. Since the original data is of
    # size 2, the weights from x and y will be batched into segments of 2 from
    # this 1d vector. For our purposes we can simply take the first index,
    # representing the weight connecting the x datapoint to the first excitatory
    # neuron.
    starting_weights_filtered_and_masked = weights[mask_expanded.bool()]
    assert starting_weights_filtered_and_masked[0] > 0.1
    assert starting_weights_filtered_and_masked[1] > 0.1

    net.process_data_online(train_data_loader)

    weights = layer.forward_weights.weight()
    weights_filtered_and_masked = weights[mask_expanded.bool()]
    print(starting_weights_filtered_and_masked)
    print(weights_filtered_and_masked)
    assert weights_filtered_and_masked[0] > 0.3
    assert weights_filtered_and_masked[1] < 0.05
