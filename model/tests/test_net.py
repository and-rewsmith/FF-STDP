import gc
import subprocess
import logging

import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from datasets.src.zenke_2a.constants import TEST_DATA_PATH, TRAIN_DATA_PATH
from datasets.src.zenke_2a.datagen import generate_sequential_dataset
from datasets.src.zenke_2a.dataset import SequentialDataset
from model.src.constants import ENCODE_SPIKE_TRAINS
from model.src.logging_util import set_logging
from model.src.network import Net
from model.src.settings import Settings

THIS_TEST_NUM_SAMPLES = 500
THIS_TEST_NUM_DATAPOINTS = 5000


# # We need to shell out to bash due to bug in pandas dataframe regeneration
# # causing severe input buffer lag
# def regenerate_data_if_needed():
#     dataframe_tmp = pd.read_csv(TRAIN_DATA_PATH)
#     samples = dataframe_tmp.groupby('sample')
#     sample_data = samples.get_group(0)
#     sample_data = sample_data[['x', 'y']].to_numpy()
#     sample_tensor = torch.tensor(sample_data, dtype=torch.float)
#     if len(dataframe_tmp) != THIS_TEST_NUM_SAMPLES or sample_tensor.shape[0] != THIS_TEST_NUM_DATAPOINTS:
#         logging.warning(
#             "Dataframe dimensions do not match the planned batch size or number of timesteps. Regenerating data...")

#         # TODOPRE: This needs to take args for dims
#         result = subprocess.run(['python', '-m', 'datasets.src.zenke_2a.datagen'], capture_output=True, text=True)
#         print(result.stdout)
#         print(result.stderr)
#         print("done")
#         input()

#         path = TRAIN_DATA_PATH
#         dataframe_tmp.to_csv(path, index=False)
#         dataframe_tmp = pd.read_csv(path)
#         del dataframe_tmp
#         del samples
#         del sample_data
#         del sample_tensor


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)
    torch.set_printoptions(precision=10, sci_mode=False)

    set_logging()

    # regenerate_data_if_needed()
    # gc.collect()

    settings = Settings(
        layer_sizes=[2],
        num_steps=THIS_TEST_NUM_DATAPOINTS,
        data_size=2,
        batch_size=THIS_TEST_NUM_SAMPLES,
        learning_rate=0.01,
        epochs=10,
        encode_spike_trains=ENCODE_SPIKE_TRAINS
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

    train_dataframe = pd.read_csv(TRAIN_DATA_PATH)
    train_sequential_dataset = SequentialDataset(
        train_dataframe, num_timesteps=settings.num_steps)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=settings.batch_size, shuffle=False)

    test_dataframe = pd.read_csv(TEST_DATA_PATH)
    test_sequential_dataset = SequentialDataset(test_dataframe, num_timesteps=settings.num_steps)
    test_data_loader = DataLoader(
        test_sequential_dataset, batch_size=10, shuffle=False)

    net = Net(settings)
    net.process_data_online(train_data_loader, test_data_loader)
