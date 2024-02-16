import math


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
        """
        self.rise: float = 0
        self.fall: float = 0
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall

    def apply(self, error: float, dt: float = 1) -> float:
        # Apply the exponential decay to the rise state and add the error
        self.rise = self.rise * math.exp(-dt / self.tau_rise) + error

        # Apply the exponential decay to the fall state and add the rise state
        self.fall = self.fall * math.exp(-dt / self.tau_fall) + self.rise

        return self.fall


class SpikeMovingAverage:

    def __init__(self, tau_mean: float) -> None:
        """
        tau_mean:
         * A time constant that determines the smoothing factor for the moving average
           of spikes.
         * A smaller tau_mean will make the average more sensitive to recent spikes.
         * A larger tau_mean will give a smoother average that is less responsive to
           individual spikes, reflecting a longer-term average rate.
        """
        self.mean: float = 0
        self.tau_mean = tau_mean

    def apply(self, spike: float, dt: float = 1) -> float:
        # Apply the exponential decay to the mean state and add the new spike value
        self.mean += (dt / self.tau_mean) * (spike - self.mean)
        return self.mean

    def tracked_value(self) -> float:
        return self.mean


class VarianceMovingAverage:

    def __init__(self, tau_var: float) -> None:
        """
        tau_var:
         * A time constant that sets the smoothing factor for the moving average
           of the variance of spikes.
         * A smaller tau_var makes the variance more sensitive to recent fluctuations.
         * A larger tau_var results in a smoother variance calculation, less affected
           by short-term changes and more reflective of long-term variability.
        """
        self.variance: float = 0
        self.tau_var: float = tau_var

    def apply(self, spike: float, spike_moving_average: float, dt: float = 1) -> float:

        # Apply the exponential decay to the variance state and add the squared deviation
        self.variance += (dt / self.tau_var) * \
            ((spike - spike_moving_average) ** 2 - self.variance)

        return self.variance

    def tracked_value(self) -> float:
        return self.variance
