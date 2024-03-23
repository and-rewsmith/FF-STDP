import random
import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src.layer import Layer
from model.src.logging_util import set_logging
from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer

THIS_TEST_NUM_SAMPLES = 1
THIS_TEST_NUM_DATAPOINTS = 1000

# TODO: use these config too
# tau_rise_alpha, tau_fall_alpha, tau_rise_epsilon, tau_fall_epsilon


def bench_many_seeds(running_log, layer_sizes, learning_rate, dt, percentage_inhibitory, exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity):
    run_settings = f"""
    running with:
    layer_sizes: {layer_sizes}
    learning_rate: {learning_rate}
    dt: {dt}
    percentage_inhibitory: {percentage_inhibitory}
    exc_to_inhib_conn_c: {exc_to_inhib_conn_c}
    exc_to_inhib_conn_sigma_squared: {exc_to_inhib_conn_sigma_squared}
    layer_sparsity: {layer_sparsity}
    """
    running_log.write(f"benching many seeds for:\n{run_settings}")
    pass_count = 0
    total_count = 0
    for _ in range(10):
        is_pass = bench_specific_configuration(running_log,
                                               layer_sizes, learning_rate, dt, percentage_inhibitory,
                                               exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity)
        if is_pass:
            pass_count += 1
        total_count += 1

    running_log.write(
        run_settings + f"\npass_rate: {pass_count / total_count}\n\n===============================================================================")


def bench_specific_configuration(running_log, layer_sizes, learning_rate, dt, percentage_inhibitory, exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity):
    rand = random.randint(1000, 9999)
    torch.manual_seed(rand)

    settings = Settings(
        layer_sizes=layer_sizes,
        data_size=2,
        batch_size=THIS_TEST_NUM_SAMPLES,
        learning_rate=learning_rate,
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
    mask = layer.excitatory_mask_vec
    mask_expanded = mask.unsqueeze(
        1).expand(-1, layer.layer_settings.data_size)
    starting_weights_filtered_and_masked = weights[mask_expanded.bool()]

    net.process_data_online(train_data_loader)

    weights = layer.forward_weights.weight()
    weights_filtered_and_masked = weights[mask_expanded.bool()]

    is_pass = weights_filtered_and_masked[0] > 0.3 and weights_filtered_and_masked[1] < 0.05

    message = f"""---------------------------------
    starting weights: {starting_weights_filtered_and_masked}
    ending weights: {weights_filtered_and_masked}
    is_pass: {is_pass}
    ---------------------------------
    """
    running_log.write(message)

    return is_pass


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=10, sci_mode=False)

    set_logging()

    running_log = open("running_log.txt", "w")
    bench_many_seeds(running_log)
