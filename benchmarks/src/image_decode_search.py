import math
import logging
import random
import time
from typing import List, TextIO

from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
from torch import nn
from profilehooks import profile

from benchmarks.tests.test_vanilla_rnn import VanillaSpikingRNN
from datasets.src.image_detection.dataset import ImageDataset
from model.src import logging_util
from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from model.src.network import Net
from model.src.settings import Settings

# TODOPRE: Think about trade off between high and 1 batch size
BATCH_SIZE = 128
DECODER_EPOCHS_PER_TRIAL = 7
DEVICE = "mps"
NUM_SEEDS_BENCH = 4
datetime_str = time.strftime("%Y%m%d-%H%M%S")
RUNNING_LOG_FILENAME = f"running_log_{datetime_str}.log"
NUM_NEURONS_CONNECT_ACROSS_LAYERS = 2


class Decoder(nn.Module):
    def __init__(self, input_size: int, num_switches: int, num_classes: int, device: str, hidden_sizes: List[int]):
        super().__init__()
        self.num_switches = num_switches
        self.num_classes = num_classes

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_switches * num_classes))
        self.layers = nn.Sequential(*layers)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_size)
        x = self.layers(x)
        # x: (batch_size, num_switches * num_classes)
        x = x.view(-1, self.num_switches, self.num_classes)
        # x: (batch_size, num_switches, num_classes)
        return torch.softmax(x, dim=-1)
        # return: (batch_size, num_switches, num_classes)

    def train(self, internal_state: torch.Tensor, labels: torch.Tensor, image_count: int, num_timesteps_each_image: int, labels_prev: torch.Tensor, num_epochs: int = DECODER_EPOCHS_PER_TRIAL):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        for _ in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.forward(internal_state)
            # outputs: (batch_size, num_switches, num_classes)

            num_images_seen = math.ceil(image_count / num_timesteps_each_image)
            labels_clone = labels.clone().detach()
            labels_clone = labels_clone[:, :num_images_seen]
            labels_prev_clone = labels_prev.clone().detach()
            labels_prev_clone = labels_prev_clone[:, num_images_seen:]
            merged_labels = torch.cat((labels_prev_clone, labels_clone), dim=1)

            # labels: (batch_size, num_images_seen)

            # Flatten outputs and labels for the loss calculation
            outputs = outputs.reshape(-1, self.num_classes)
            # outputs: (batch_size * num_switches, num_classes)
            merged_labels = merged_labels.reshape(-1)
            # labels: (batch_size * num_switches,)

            loss = criterion(outputs, merged_labels)
            wandb.log({"loss": loss.item()})
            # loss: scalar

            loss.backward()
            optimizer.step()

    def predict(self, internal_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(internal_state)
            # outputs: (batch_size, num_switches, num_classes)
            _, predicted = torch.max(outputs, dim=-1)
            # predicted: (batch_size, num_switches)
            return predicted


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
    exc_to_inhib_conn_c = wandb.config.exc_to_inhib_conn_c
    exc_to_inhib_conn_sigma_squared = wandb.config.exc_to_inhib_conn_sigma_squared
    percentage_inhibitory = wandb.config.percentage_inhibitory
    decay_beta = wandb.config.decay_beta
    tau_mean = wandb.config.tau_mean
    tau_var = wandb.config.tau_var
    tau_stdp = wandb.config.tau_stdp
    tau_rise_alpha = wandb.config.tau_rise_alpha
    tau_fall_alpha = wandb.config.tau_fall_alpha
    tau_rise_epsilon = wandb.config.tau_rise_epsilon
    tau_fall_epsilon = wandb.config.tau_fall_epsilon
    threshold_scale = wandb.config.threshold_scale
    threshold_decay = wandb.config.threshold_decay

    # sum layer sizes to get total neurons
    total_neurons = sum(layer_sizes)
    layer_sparsity = NUM_NEURONS_CONNECT_ACROSS_LAYERS / total_neurons

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
    threshold_scale: {threshold_scale},
    threshold_decay: {threshold_decay},
    """
    logging.info(run_settings)

    with open(RUNNING_LOG_FILENAME, "a") as running_log:
        running_log.write(f"{run_settings}")
        running_log.flush()

        cum_pass_rate = 0
        for _ in range(NUM_SEEDS_BENCH):
            pass_rate = bench_specific_seed(
                running_log,
                layer_sizes, learning_rate, dt, percentage_inhibitory,
                exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity,
                decay_beta, tau_mean, tau_var, tau_stdp, tau_rise_alpha, tau_fall_alpha,
                tau_rise_epsilon, tau_fall_epsilon, threshold_scale, threshold_decay
            )
            wandb.log({"image_predict_success": pass_rate})
            cum_pass_rate += pass_rate

        running_log.write(
            run_settings + f"\average_image_predict_success: {cum_pass_rate / NUM_SEEDS_BENCH}\n\n======================================\
                =========================================")
        running_log.flush()

    wandb.log({"average_image_predict_success": cum_pass_rate / NUM_SEEDS_BENCH})


@profile(stdout=False, filename='baseline.prof',
         skip=True)
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
                        tau_fall_epsilon: float,
                        threshold_scale: float,
                        threshold_decay: float) -> float:
    rand = random.randint(1000, 9999)
    torch.manual_seed(rand)
    running_log.write(f"Seed: {rand}\n")

    dataset = ImageDataset(
        num_timesteps_each_image=20,
        num_switches=5,
        device=DEVICE
    )
    settings = Settings(
        layer_sizes=layer_sizes,
        data_size=dataset.num_classes,
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
        threshold_scale=threshold_scale,
        threshold_decay=threshold_decay,
        device=torch.device(DEVICE)
    )

    train_dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    net = VanillaSpikingRNN(settings).to(settings.device)
    decoder = Decoder(input_size=sum(layer_sizes), num_switches=dataset.num_switches,
                      num_classes=dataset.num_classes, device=DEVICE, hidden_sizes=[100, 50, 20])

    prev_labels = None
    batch_count = 0
    for batch, labels in tqdm(train_dataloader):
        labels_untouched = labels.clone().detach()
        # batch: (num_timesteps, batch_size, data_size)
        # labels: (batch_size,)
        batch = batch.permute(1, 0, 2)
        # batch: (batch_size, num_timesteps, data_size)
        timestep_count = 0
        for timestep_data in tqdm(batch, leave=False):
            timestep_count += 1  # position here matters: predict current image rather than one timestep ago
            # timestep_data: (batch_size, data_size)
            net.process_data_single_timestep(timestep_data)

            if batch_count >= 1:
                layer_activations = net.layer_activations()
                # layer_activations: List[torch.Tensor], each tensor of shape (batch_size, layer_size)
                internal_state = torch.concat(layer_activations, dim=1)
                # internal_state: (batch_size, sum(layer_sizes))

                # Reshape labels to (batch_size, num_switches)
                labels = labels.view(-1, dataset.num_switches)

                decoder.train(internal_state, labels, timestep_count, dataset.num_timesteps_each_image, prev_labels)

        prev_labels = labels_untouched
        batch_count += 1

    dataset = ImageDataset(
        num_timesteps_each_image=20,
        num_switches=5,
        device=DEVICE,
        max_samples=1024
    )
    test_dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    total_correct = 0
    total = 0
    for batch, labels in tqdm(test_dataloader):
        # batch: (num_timesteps, batch_size, data_size)
        # labels: (batch_size,)
        batch = batch.permute(1, 0, 2)
        # batch: (batch_size, num_timesteps, data_size)
        for timestep_data in tqdm(batch, leave=False):
            # timestep_data: (batch_size, data_size)
            net.process_data_single_timestep(timestep_data)

        layer_activations = net.layer_activations()
        # layer_activations: List[torch.Tensor], each tensor of shape (batch_size, layer_size)
        internal_state = torch.concat(layer_activations, dim=1)
        # internal_state: (batch_size, sum(layer_sizes))

        predictions = decoder.predict(internal_state)
        # predictions: (batch_size, num_switches)

        num_correct = torch.sum(predictions == labels).item()
        total_correct += num_correct
        total += labels.numel()

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

    running_log = open(RUNNING_LOG_FILENAME, "w")
    message = f"Sweep logs. Current datetime: {time.ctime()}\n"
    running_log.write(message)
    running_log.close()
    logging.debug(message)

    # sweep_configuration = {
    #     "method": "bayes",
    #     "metric": {"goal": "maximize", "name": "average_image_predict_success"},
    #     "parameters": {
    #         "layer_sizes": {"values": [[75, 75, 75, 75], [100, 100, 100, 100], [200, 200, 200, 200], [400, 400, 400, 400], [600, 600, 600, 600]]},
    #         "learning_rate": {"min": 0.0001, "max": 0.01},
    #         "dt": {"min": 0.001, "max": 1.0},
    #         "percentage_inhibitory": {"min": 10, "max": 60},
    #         "exc_to_inhib_conn_c": {"min": 0.25, "max": 0.75},
    #         "exc_to_inhib_conn_sigma_squared": {"min": 1, "max": 60},
    #         "layer_sparsity": {"min": 0.1, "max": 0.9},
    #         "tau_mean": {"min": 30, "max": 1800},
    #         "tau_var": {"min": 0.01, "max": 0.1},
    #         "tau_stdp": {"min": 0.01, "max": 0.1},
    #         "tau_rise_alpha": {"min": 0.001, "max": 0.01},
    #         "tau_fall_alpha": {"min": 0.005, "max": 0.05},
    #         "tau_rise_epsilon": {"min": 0.002, "max": 0.02},
    #         "tau_fall_epsilon": {"min": 0.01, "max": 0.1},
    #         "decay_beta": {"min": 0.8, "max": 0.95},
    #         "threshold_scale": {"min": 1.0, "max": 1.5},
    #         "threshold_decay": {"min": 0.9, "max": 1.0},
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="LPL-SNN-4")
    sweep_id = "and-rewsmith/FF-STDP-benchmarks_src/oh7quwpm"

    wandb.agent(sweep_id, function=objective)
