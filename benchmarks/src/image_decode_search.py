import logging
import random
import time
from typing import Any, TextIO

import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader
from datasets.src.image_detection.dataset import ImageDataset

from model.src import logging_util
from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src.layer import Layer
from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer

BATCH_SIZE = 256


def objective() -> None:
    wandb.init(
        project="LPL-SNN-4",
        config={
            "architecture": "initial",
            "dataset": "point-cloud",
        },
        allow_val_change=True  # TODOPRE: review this as it is silencing a warning
    )

    layer_sizes = wandb.config.layer_sizes
    learning_rate = wandb.config.learning_rate
    dt = wandb.config.dt
    percentage_inhibitory = wandb.config.percentage_inhibitory
    exc_to_inhib_conn_c = wandb.config.exc_to_inhib_conn_c
    exc_to_inhib_conn_sigma_squared = wandb.config.exc_to_inhib_conn_sigma_squared
    layer_sparsity = wandb.config.layer_sparsity
    decay_beta = wandb.config.decay_beta
    tau_mean = wandb.config.tau_mean
    tau_var = wandb.config.tau_var
    tau_stdp = wandb.config.tau_stdp
    tau_rise_alpha = wandb.config.tau_rise_alpha
    tau_fall_alpha = wandb.config.tau_fall_alpha
    tau_rise_epsilon = wandb.config.tau_rise_epsilon
    tau_fall_epsilon = wandb.config.tau_fall_epsilon

    run_settings = f"""
    running with:
    layer_sizes: {layer_sizes}
    learning_rate: {learning_rate}
    dt: {dt}
    percentage_inhibitory: {percentage_inhibitory}
    exc_to_inhib_conn_c: {exc_to_inhib_conn_c}
    exc_to_inhib_conn_sigma_squared: {exc_to_inhib_conn_sigma_squared}
    layer_sparsity: {layer_sparsity}
    decay_beta: {decay_beta},
    tau_mean: {tau_mean},
    tau_var: {tau_var},
    tau_stdp: {tau_stdp},
    tau_rise_alpha: {tau_rise_alpha},
    tau_fall_alpha: {tau_fall_alpha},
    tau_rise_epsilon: {tau_rise_epsilon},
    tau_fall_epsilon: {tau_fall_epsilon},
    """
    logging.info(run_settings)

    with open("running_log.log", "a") as running_log:
        running_log.write(f"{run_settings}")
        running_log.flush()

        cum_pass_rate = 0
        num_seeds_bench = 10
        for _ in range(num_seeds_bench):
            pass_rate = bench_specific_seed(
                running_log,
                layer_sizes, learning_rate, dt, percentage_inhibitory,
                exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity,
                decay_beta, tau_mean, tau_var, tau_stdp, tau_rise_alpha, tau_fall_alpha,
                tau_rise_epsilon, tau_fall_epsilon
            )
            cum_pass_rate += pass_rate
            total_count += 1

        running_log.write(
            run_settings + f"\average_image_predict_success: {cum_pass_rate / num_seeds_bench}\n\n======================================\
                =========================================")
        running_log.flush()

    wandb.log({"average_image_predict_success": cum_pass_rate / num_seeds_bench})


def bench_specific_seed(running_log: TextIO,
                        layer_sizes: list[int],
                        learning_rate: float,
                        dt: float,
                        percentage_inhibitory: int,
                        exc_to_inhib_conn_c: float,
                        exc_to_inhib_conn_sigma_squared: float,
                        layer_sparsity: float,
                        decay_beta: float,
                        tau_mean: float,
                        tau_var: float,
                        tau_stdp: float,
                        tau_rise_alpha: float,
                        tau_fall_alpha: float,
                        tau_rise_epsilon: float,
                        tau_fall_epsilon: float) -> bool:
    rand = random.randint(1000, 9999)
    torch.manual_seed(rand)

    settings = Settings(
        layer_sizes=layer_sizes,
        data_size=2,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        epochs=10,
        encode_spike_trains=ENCODE_SPIKE_TRAINS,
        dt=dt,
        percentage_inhibitory=percentage_inhibitory,
        exc_to_inhib_conn_c=exc_to_inhib_conn_c,
        exc_to_inhib_conn_sigma_squared=exc_to_inhib_conn_sigma_squared,
        layer_sparsity=layer_sparsity,
        decay_beta=decay_beta,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_stdp=tau_stdp,
        tau_rise_alpha=tau_rise_alpha,
        tau_fall_alpha=tau_fall_alpha,
        tau_rise_epsilon=tau_rise_epsilon,
        tau_fall_epsilon=tau_fall_epsilon,
        device=torch.device("cpu")
    )

    dataset = ImageDataset(
        num_timesteps_each_image=20,
        num_timesteps_flash=20,
        num_switches=5,
        switch_probability=0.25
    )
    train_dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    net = Net(settings).to(settings.device)
    decoder = Decoder()

    for batch, labels in train_dataloader:
        batch = batch.permute(1, 0, 2)
        for timestep_data in batch:
            net.process_data_single_timestep(timestep_data)

        layer_activations = net.layer_activations()
        internal_state = torch.concat(layer_activations, dim=1)

        decoder.train(internal_state, labels)

        net = Net(settings).to(settings.device)

    dataset = ImageDataset(
        num_timesteps_each_image=20,
        num_timesteps_flash=20,
        num_switches=5,
        switch_probability=0.25
    )
    test_dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    total_correct = 0
    total = 0
    for batch, labels in test_dataloader:
        batch = batch.permute(1, 0, 2)
        for timestep_data in batch:
            net.process_data_single_timestep(timestep_data)

        layer_activations = net.layer_activations()
        internal_state = torch.concat(layer_activations, dim=1)

        predictions = decoder.forward(internal_state, labels)
        num_correct = torch.sum(predictions == labels).item()
        total_correct += num_correct
        total += len(labels)

    pass_rate = total_correct / total

    message = f"""---------------------------------
    pass_rate: {pass_rate}
    ---------------------------------
    """
    running_log.write(message)
    running_log.flush()
    logging.info(message)

    return pass_rate


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=10, sci_mode=False)
    logging_util.set_logging()

    running_log = open("running_log.log", "w")
    message = f"Sweep logs. Current datetime: {time.ctime()}\n"
    running_log.write(message)
    running_log.close()
    logging.debug(message)

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "pass_rate"},
        "parameters": {
            "layer_sizes": {"values": [[20, 20, 20, 20], [40, 40, 40, 40], [60, 60, 60, 60], [100, 100, 100, 100]]},
            "learning_rate": {"min": 0.0001, "max": 0.01},
            "dt": {"min": 0.001, "max": 1.0},
            "percentage_inhibitory": {"min": 30, "max": 60},
            "exc_to_inhib_conn_c": {"min": 0.25, "max": 0.75},
            "exc_to_inhib_conn_sigma_squared": {"min": 1, "max": 60},
            "layer_sparsity": {"min": 0.1, "max": 0.9},
            "tau_mean": {"min": 30, "max": 1800},
            "tau_var": {"min": 0.01, "max": 0.1},
            "tau_stdp": {"min": 0.01, "max": 0.1},
            "tau_rise_alpha": {"min": 0.001, "max": 0.01},
            "tau_fall_alpha": {"min": 0.005, "max": 0.05},
            "tau_rise_epsilon": {"min": 0.002, "max": 0.02},
            "tau_fall_epsilon": {"min": 0.01, "max": 0.1},
            "decay_beta": {"min": 0.8, "max": 0.9},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="LPL-SNN-2")
    wandb.agent(sweep_id, function=objective)
