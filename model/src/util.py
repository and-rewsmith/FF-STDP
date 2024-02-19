import math
from collections import deque
from typing import Any, Deque, Optional, Tuple

import snntorch as snn
import torch

MAX_RETAINED_SPIKES = 2

# Zenke's paper uses a tau_mean of 600s
TAU_MEAN = 600
# Zenke's paper uses a tau_var of 20ms
TAU_VAR = .02


class MovingAverageLIF(snn.Leaky):
    def __init__(self, *args: Any, batch_size: int, layer_size: int, tau_mean: float = TAU_MEAN,
                 tau_var: float = TAU_VAR, **kwargs: Any) -> None:
        super(MovingAverageLIF, self).__init__(*args, **kwargs)
        self.spike_moving_average = SpikeMovingAverage(
            tau_mean=tau_mean, batch_size=batch_size, data_size=layer_size)
        self.variance_moving_average = VarianceMovingAverage(tau_var=tau_var)

    def forward(self, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        forward_output = super(
            MovingAverageLIF, self).forward(current, mem)
        spike = forward_output[0]
        mem = forward_output[1]

        mean_spike = self.spike_moving_average.apply(spike)
        self.variance_moving_average.apply(spike, mean_spike)

        return (spike, mem)

    def tracked_spike_moving_average(self) -> torch.Tensor:
        return self.spike_moving_average.tracked_value()

    def tracked_variance_moving_average(self) -> torch.Tensor:
        return self.variance_moving_average.tracked_value()


class TemporalFilter:

    def __init__(self, tau_rise: float, tau_fall: float) -> None:
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

    def apply(self, value: torch.Tensor, dt: float = 1) -> torch.Tensor:
        if self.rise is None:
            # Initialize rise based on the first error received
            self.rise = torch.zeros_like(value)

        if self.fall is None:
            # Initialize fall based on the first error received
            self.fall = torch.zeros_like(value)

        # Apply the exponential decay to the rise state and add the error
        self.rise = self.rise * math.exp(-dt / self.tau_rise) + value

        # Apply the exponential decay to the fall state and add the rise state
        self.fall = self.fall * math.exp(-dt / self.tau_fall) + self.rise

        return self.fall


class SpikeMovingAverage:

    def __init__(self, batch_size: int, data_size: int, tau_mean: float = TAU_MEAN) -> None:
        """
        tau_mean:
         * A time constant that determines the smoothing factor for the moving average
           of spikes.
         * A smaller tau_mean will make the average more sensitive to recent spikes.
         * A larger tau_mean will give a smoother average that is less responsive to
           individual spikes, reflecting a longer-term average rate.

        Zenke's paper uses a tau_mean of 600s.

        TODO: Concern here is that we are initializing the mean state with zeros,
                which might not be the best approach.
        """
        self.mean: Optional[torch.Tensor] = None
        self.tau_mean = tau_mean
        self.spike_rec: Deque[torch.Tensor] = deque(maxlen=MAX_RETAINED_SPIKES)
        for _ in range(2):
            self.spike_rec.append(torch.zeros(
                batch_size, data_size))

    def apply(self, spike: torch.Tensor, dt: float = 1) -> torch.Tensor:
        self.spike_rec.append(spike)

        if self.mean is None:
            # Initialize mean based on the first spike received
            self.mean = torch.zeros_like(spike)

        # Apply the exponential decay to the mean state and add the new spike value
        self.mean += (dt / self.tau_mean) * (spike - self.mean)
        return self.mean

    def tracked_value(self) -> torch.Tensor:
        if self.mean is None:
            raise ValueError("No data has been received yet")

        return self.mean


class VarianceMovingAverage:

    def __init__(self, tau_var: float = TAU_VAR) -> None:
        """
        tau_var:
         * A time constant that sets the smoothing factor for the moving average
           of the variance of spikes.
         * A smaller tau_var makes the variance more sensitive to recent fluctuations.
         * A larger tau_var results in a smoother variance calculation, less affected
           by short-term changes and more reflective of long-term variability.

        Zenke's paper uses a tau_var of 20ms.

        TODO: Concern here is that we are initializing the variance state with zeros,
                which might not be the best approach.
        """
        self.variance: Optional[torch.Tensor] = None
        self.tau_var: float = tau_var

    def apply(self, spike: torch.Tensor, spike_moving_average: torch.Tensor, dt: float = 1) -> torch.Tensor:
        if self.variance is None:
            # Initialize variance based on the first spike received
            self.variance = torch.zeros_like(spike)

        # Apply the exponential decay to the variance state and add the squared deviation
        self.variance += (dt / self.tau_var) * \
            ((spike - spike_moving_average) ** 2 - self.variance)

        return self.variance

    def tracked_value(self) -> torch.Tensor:
        if self.variance is None:
            raise ValueError("No data has been received yet")

        return self.variance
