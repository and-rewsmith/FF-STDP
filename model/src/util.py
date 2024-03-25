from enum import Enum
import math
from collections import deque
from typing import Deque, Optional, Tuple

import torch

from model.src.constants import DT, MAX_RETAINED_SPIKES, TAU_MEAN, \
    TAU_STDP, TAU_VAR

from model.src.lif import LIF
from model.src.settings import LayerSettings


class SynapticUpdateType(Enum):
    RECURRENT = 1
    FORWARD = 2
    BACKWARD = 3


class MovingAverageLIF:
    def __init__(self, layer_settings: LayerSettings) -> None:
        self.spike_moving_average = SpikeMovingAverage(
            tau_mean=layer_settings.tau_mean, batch_size=layer_settings.batch_size,
            data_size=layer_settings.size, device=layer_settings.device)
        self.variance_moving_average = VarianceMovingAverage(
            tau_var=layer_settings.tau_var, device=layer_settings.device)
        self.neuron_layer = LIF(layer_settings.decay_beta)
        self.dt = layer_settings.dt

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        spike = self.neuron_layer.forward(current)

        mean_spike = self.spike_moving_average.apply(spike, self.dt)
        self.variance_moving_average.apply(spike, mean_spike, self.dt)

        return (spike)

    def mem(self) -> torch.Tensor:
        if self.neuron_layer.prereset_mem is None:
            raise ValueError("No data has been received yet")

        # NOTE: The membrane potential returned here is used in the learning
        # rule. Currently, we are not returning what Zenke's paper uses (they
        # use `neuron_layer.mem`).
        return self.neuron_layer.prereset_mem

    def tracked_spike_moving_average(self) -> torch.Tensor:
        return self.spike_moving_average.tracked_value()

    def tracked_variance_moving_average(self) -> torch.Tensor:
        return self.variance_moving_average.tracked_value()


class DoubleExponentialFilter:

    def __init__(self, tau_rise: float, tau_fall: float,
                 device: torch.device) -> None:
        """
        tau_rise:
         * This controls how quickly the filter responds to an increase in the
           prediction error.
         * A smaller tau_rise means a faster response to changes, making the
           filter more sensitive to sudden increases in error.
         * A larger tau_rise will smooth out the response, making the filter
           less sensitive to abrupt changes.

        tau_fall:
         * This parameter determines the decay rate of the filter's response.
         * A smaller tau_fall will make the filter's response to decreases in
           the error signal more rapid.
         * A larger tau_fall will result in a slower decay, allowing the filter
           to retain the influence of past errors for longer.

        If you're dealing with high-frequency noise, you might start with a
        tau_rise that's quite small (fractions of the time step) to quickly
        adapt to changes, while tau_fall could be larger to prevent the filter
        from reacting too strongly to what might be noise. Conversely, for a
        system with slow-moving dynamics, you might choose larger values for
        both tau_rise and tau_fall to avoid reacting to insignificant changes.

        Zenke's paper uses:
         * alpha: tau_rise of 2ms and a tau_fall of 10ms
         * epsilon: tau_rise of 5ms and a tau_fall of 20ms

         TODO: Concern here is that we are initializing the rise and fall states
                with zeros, which might not be the best approach.
        """
        self.rise: Optional[torch.Tensor] = None
        self.fall: Optional[torch.Tensor] = None
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.device = device

    def apply(self, value: torch.Tensor, dt: float = DT) -> torch.Tensor:
        if self.rise is None:
            # Initialize rise based on the first error received
            self.rise = torch.zeros_like(value).to(self.device)

        if self.fall is None:
            # Initialize fall based on the first error received
            self.fall = torch.zeros_like(value).to(self.device)

        # Apply the exponential decay to the rise state and add the error
        decay_factor_rise = math.exp(-dt / self.tau_rise)
        self.rise = self.rise * decay_factor_rise + value

        # Apply the exponential decay to the fall state and add the rise state
        decay_factor_fall = math.exp(-dt / self.tau_fall)
        self.fall = self.fall * decay_factor_fall + \
            (1 - decay_factor_fall) * self.rise

        return self.fall


class ExcitatorySynapseFilterGroup:

    def __init__(self, layer_settings: LayerSettings) -> None:

        self.first_term_alpha = DoubleExponentialFilter(
            layer_settings.tau_rise_alpha, layer_settings.tau_fall_alpha, device=layer_settings.device)
        self.first_term_epsilon = DoubleExponentialFilter(
            layer_settings.tau_rise_epsilon, layer_settings.tau_fall_epsilon, device=layer_settings.device)
        self.second_term_alpha = DoubleExponentialFilter(
            layer_settings.tau_rise_alpha, layer_settings.tau_fall_alpha, device=layer_settings.device)


class SpikeMovingAverage:

    def __init__(self, batch_size: int, data_size: int,
                 device: torch.device, tau_mean: float = TAU_MEAN) -> None:
        """
        tau_mean:
         * A time constant that determines the smoothing factor for the moving average
           of spikes.
         * A smaller tau_mean will make the average more sensitive to recent spikes.
         * A larger tau_mean will give a smoother average that is less responsive to
           individual spikes, reflecting a longer-term average rate.

        Zenke's paper uses a tau_mean of 600s.

        TODO: Concern here is that we are initializing the state with zeros,
                which might not be the best approach.
        """
        self.device = device
        self.mean: Optional[torch.Tensor] = None
        self.tau_mean = tau_mean
        self.spike_rec: Deque[torch.Tensor] = deque(maxlen=MAX_RETAINED_SPIKES)
        for _ in range(MAX_RETAINED_SPIKES):
            self.spike_rec.append(torch.zeros(
                batch_size, data_size).to(device=device))

    def apply(self, spike: torch.Tensor, dt: float = DT) -> torch.Tensor:
        self.spike_rec.append(spike)

        if self.mean is None:
            # Initialize mean based on the first spike received
            self.mean = torch.zeros_like(spike).to(self.device)

        # Apply the exponential decay to the mean state and add the new spike
        # value
        decay_factor = math.exp(-dt / self.tau_mean)
        self.mean = self.mean * decay_factor + (1 - decay_factor) * spike

        return self.mean

    def tracked_value(self) -> torch.Tensor:
        if self.mean is None:
            raise ValueError("No data has been received yet")

        return self.mean


class VarianceMovingAverage:

    def __init__(self, device: torch.device, tau_var: float = TAU_VAR) -> None:
        """
        tau_var:
         * A time constant that sets the smoothing factor for the moving average
           of the variance of spikes.
         * A smaller tau_var makes the variance more sensitive to recent fluctuations.
         * A larger tau_var results in a smoother variance calculation, less affected
           by short-term changes and more reflective of long-term variability.

        Zenke's paper uses a tau_var of 20ms.

        TODO: Concern here is that we are initializing state with zeros,
                which might not be the best approach.
        """
        self.variance: Optional[torch.Tensor] = None
        self.tau_var: float = tau_var
        self.device = device

    def apply(self, spike: torch.Tensor, spike_moving_average: torch.Tensor,
              dt: float = DT) -> torch.Tensor:
        if self.variance is None:
            # Initialize variance based on the first spike received
            self.variance = torch.zeros_like(spike).to(self.device)

        # Apply the exponential decay to the variance state and add the squared
        # deviation
        decay_factor = math.exp(-dt / self.tau_var)
        self.variance = self.variance * decay_factor + \
            (1 - decay_factor) * (spike - spike_moving_average) ** 2

        assert self.variance is not None
        return self.variance

    def tracked_value(self) -> torch.Tensor:
        if self.variance is None:
            raise ValueError("No data has been received yet")

        return self.variance


class InhibitoryPlasticityTrace:

    def __init__(self, trace_shape: Tuple[int, int],
                 device: torch.device, tau_stdp: float = TAU_STDP) -> None:
        """
        Zenke's paper uses a tau_stdp of 20ms.

        TODO: Concern here is that we are initializing the state with zeros,
                which might not be the best approach.
        """
        self.device = device
        self.trace = torch.zeros(trace_shape).to(device)
        self.tau_stdp: float = tau_stdp

    def apply(self, spike: torch.Tensor, dt: float = DT) -> torch.Tensor:
        decay_factor = math.exp(-dt / self.tau_stdp)
        self.trace = self.trace * decay_factor + spike

        return self.trace

    def tracked_value(self) -> torch.Tensor:
        return self.trace
