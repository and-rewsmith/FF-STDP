import torch

from model.src.constants import DECAY_BETA, DT, EXC_TO_INHIB_CONN_C, EXC_TO_INHIB_CONN_SIGMA_SQUARED, LAYER_SPARSITY, PERCENTAGE_INHIBITORY, TAU_FALL_ALPHA, TAU_FALL_EPSILON, TAU_MEAN, TAU_RISE_ALPHA, TAU_RISE_EPSILON, TAU_STDP, TAU_VAR, THRESHOLD_SCALE, THRESHOLD_DECAY


class Settings:
    def __init__(self,
                 layer_sizes: list[int],
                 data_size: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 encode_spike_trains: bool,
                 dt: float = DT,
                 percentage_inhibitory: int = PERCENTAGE_INHIBITORY,
                 exc_to_inhib_conn_c: float = EXC_TO_INHIB_CONN_C,
                 exc_to_inhib_conn_sigma_squared: float = EXC_TO_INHIB_CONN_SIGMA_SQUARED,
                 layer_sparsity: float = LAYER_SPARSITY,
                 decay_beta: float = DECAY_BETA,
                 threshold_scale: float = THRESHOLD_SCALE,
                 threshold_decay: float = THRESHOLD_DECAY,
                 tau_mean: float = TAU_MEAN,
                 tau_var: float = TAU_VAR,
                 tau_stdp: float = TAU_STDP,
                 tau_rise_alpha: float = TAU_RISE_ALPHA,
                 tau_fall_alpha: float = TAU_FALL_ALPHA,
                 tau_rise_epsilon: float = TAU_RISE_EPSILON,
                 tau_fall_epsilon: float = TAU_FALL_EPSILON,
                 device: torch.device = torch.device("cpu")) -> None:
        self.layer_sizes = layer_sizes
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.encode_spike_trains = encode_spike_trains
        self.dt = dt
        self.percentage_inhibitory = percentage_inhibitory
        self.exc_to_inhib_conn_c = exc_to_inhib_conn_c
        self.exc_to_inhib_conn_sigma_squared = exc_to_inhib_conn_sigma_squared
        self.layer_sparsity = layer_sparsity
        self.decay_beta = decay_beta
        self.threshold_scale = threshold_scale
        self.threshold_decay = threshold_decay
        self.tau_mean = tau_mean
        self.tau_var = tau_var
        self.tau_stdp = tau_stdp
        self.tau_rise_alpha = tau_rise_alpha
        self.tau_fall_alpha = tau_fall_alpha
        self.tau_rise_epsilon = tau_rise_epsilon
        self.tau_fall_epsilon = tau_fall_epsilon
        self.device = device


class LayerSettings:
    def __init__(self,
                 layer_id: int,
                 prev_size: int,
                 size: int,
                 next_size: int,
                 batch_size: int,
                 learning_rate: float,
                 data_size: int,
                 dt: float,
                 percentage_inhibitory: int,
                 exc_to_inhib_conn_c: float,
                 exc_to_inhib_conn_sigma_squared: float,
                 layer_sparsity: float,
                 decay_beta: float,
                 threshold_scale: float,
                 threshold_decay: float,
                 tau_mean: float,
                 tau_var: float,
                 tau_stdp: float,
                 tau_rise_alpha: float,
                 tau_fall_alpha: float,
                 tau_rise_epsilon: float,
                 tau_fall_epsilon: float,
                 device: torch.device) -> None:
        self.layer_id = layer_id
        self.prev_size = prev_size
        self.size = size
        self.next_size = next_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_size = data_size
        self.dt = dt
        self.percentage_inhibitory = percentage_inhibitory
        self.exc_to_inhib_conn_c = exc_to_inhib_conn_c
        self.exc_to_inhib_conn_sigma_squared = exc_to_inhib_conn_sigma_squared
        self.layer_sparsity = layer_sparsity
        self.decay_beta = decay_beta
        self.threshold_scale = threshold_scale
        self.threshold_decay = threshold_decay
        self.tau_mean = tau_mean
        self.tau_var = tau_var
        self.tau_stdp = tau_stdp
        self.tau_rise_alpha = tau_rise_alpha
        self.tau_fall_alpha = tau_fall_alpha
        self.tau_rise_epsilon = tau_rise_epsilon
        self.tau_fall_epsilon = tau_fall_epsilon
        self.device = device
