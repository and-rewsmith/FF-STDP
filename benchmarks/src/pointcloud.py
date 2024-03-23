import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src import logging_util
from model.src.constants import LEARNING_RATE
from model.src.settings import Settings
from model.src.network import Net

NUM_STEPS = 5000
BATCH_SIZE = 500
ENCODE_SPIKE_TRAINS = False


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)
    torch.set_printoptions(precision=10, sci_mode=False)

    logging_util.set_logging()

    settings = Settings(
        layer_sizes=[2],
        data_size=2,
        batch_size=BATCH_SIZE,
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

    train_sequential_dataset = SequentialDataset(
        DatasetType.TRAIN,
        train_dataframe, num_timesteps=NUM_STEPS, planned_batch_size=settings.batch_size)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=settings.batch_size, shuffle=False)

    net = Net(settings).to(device=settings.device)
    net.process_data_online(train_data_loader)
