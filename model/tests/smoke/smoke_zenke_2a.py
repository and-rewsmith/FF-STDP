import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from datasets.src.zenke_2a.constants import TEST_DATA_PATH, TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src.layer import Layer
from model.src.logging_util import set_logging
from model.src.network import Net
from model.src.settings import Settings

THIS_TEST_NUM_SAMPLES = 100
THIS_TEST_NUM_DATAPOINTS = 3000


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)
    torch.set_printoptions(precision=10, sci_mode=False)

    set_logging()

    settings = Settings(
        layer_sizes=[2],
        num_steps=THIS_TEST_NUM_DATAPOINTS,
        data_size=2,
        batch_size=THIS_TEST_NUM_SAMPLES,
        learning_rate=0.01,
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
                                                 train_dataframe, num_timesteps=settings.num_steps,
                                                 planned_batch_size=settings.batch_size)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=settings.batch_size, shuffle=False)

    try:
        test_dataframe = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        test_dataframe = None
    test_sequential_dataset = SequentialDataset(
        DatasetType.TEST, test_dataframe, num_timesteps=settings.num_steps, planned_batch_size=settings.batch_size)
    test_data_loader = DataLoader(
        test_sequential_dataset, batch_size=10, shuffle=False)

    net = Net(settings).to(settings.device)

    # figure out what the excitatory mask is, find that neuron, then find the weights for that neuron
    layer: Layer = net.layers[0]
    weights = layer.forward_weights.weight
    mask = layer.excitatory_mask_vec
    mask_expanded = mask.unsqueeze(1).expand(-1, layer.layer_settings.data_size)
    weights_filtered_and_masked = weights[mask_expanded.bool()]
    assert weights_filtered_and_masked[0] > 0.3
    assert weights_filtered_and_masked[1] > 0.3

    net.process_data_online(train_data_loader, test_data_loader)

    weights_filtered_and_masked = weights[mask_expanded.bool()]
    assert weights_filtered_and_masked[0] > 0.3
    assert weights_filtered_and_masked[1] < 0.01
