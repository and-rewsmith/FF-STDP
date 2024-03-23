import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src.constants import LEARNING_RATE
from model.src.layer import Layer
from model.src.logging_util import set_logging
from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer

THIS_TEST_NUM_SAMPLES = 100
THIS_TEST_NUM_DATAPOINTS = 7000

"""
TODO:
check the logic for weight filtering to see that the asserts are checking the correct weight
stop spike encoding
instead of normal sampling y, make y be random
instead of sampling the x, make x be slowly varying

remove clamping

connectivity patterns
"""


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1128)
    torch.set_printoptions(precision=10, sci_mode=False)

    set_logging()

    settings = Settings(
        layer_sizes=[2, 8, 8],
        data_size=2,
        batch_size=THIS_TEST_NUM_SAMPLES,
        learning_rate=LEARNING_RATE,
        epochs=10,
        encode_spike_trains=ENCODE_SPIKE_TRAINS,
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
    visualizer = NetworkVisualizer(net)
    visualizer.update()

    # figure out what the excitatory mask is, find that neuron, then find the
    # weights for that neuron
    layer: Layer = net.layers[0]
    weights = layer.forward_weights.weight()
    print(net.layers[1].backward_weights.weight())
    mask = layer.excitatory_mask_vec
    print(mask)
    mask_expanded = mask.unsqueeze(
        1).expand(-1, layer.layer_settings.data_size)
    print(mask_expanded)
    weights_filtered_and_masked = weights[mask_expanded.bool()]
    print(weights_filtered_and_masked)
    assert weights_filtered_and_masked[0] > 0.1
    assert weights_filtered_and_masked[1] > 0.1

    net.process_data_online(train_data_loader)

    weights = layer.forward_weights.weight()
    weights_filtered_and_masked = weights[mask_expanded.bool()]
    print(weights_filtered_and_masked)
    assert weights_filtered_and_masked[0] > 0.3
    assert weights_filtered_and_masked[1] < 0.05
