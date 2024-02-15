import pytest

from model.util import SpikeMovingAverage, TemporalFilter, VarianceMovingAverage


def test_temporal_filter() -> None:
    tf = TemporalFilter(tau_rise=1, tau_fall=1)

    # Initial apply since temporal filter is second order
    tf.apply(error=1)

    # Apply a single error with a small dt should result in small change
    assert tf.apply(error=1) == pytest.approx(1.7357588823428847)

    # Apply a zero error should result in decay of the state
    assert tf.apply(error=0) == pytest.approx(1.1417647320527227)


def test_spike_moving_average() -> None:
    sma = SpikeMovingAverage(tau_mean=1)

    # Apply a single spike
    assert sma.apply(spike=1) == 1.0

    # Apply another spike, the average should increase
    assert sma.apply(spike=2) == 2.0

    # After some time with no spikes, the average should decay
    assert sma.apply(spike=0) == 0.0


def test_variance_moving_average() -> None:
    tau_mean = 10
    tau_var = 10
    sma = SpikeMovingAverage(tau_mean)
    vma = VarianceMovingAverage(tau_var)

    # Apply spikes to the moving average
    spike = 2
    for _ in range(20):
        # NOTE: The spike moving average is always updated first
        sma.apply(spike=spike)
        vma.apply(spike=spike, spike_moving_average=sma.tracked_value())
    assert sma.tracked_value() > 1

    sma_value = sma.apply(spike=spike)
    expected_variance = 0.3508073062162211
    assert vma.apply(
        spike=spike, spike_moving_average=sma_value) == pytest.approx(expected_variance)

    # After some time with no spikes, the variance should increase
    spike_2 = 0
    sma_value = sma.apply(spike=spike_2)  # Mean should decay
    assert vma.apply(
        spike=spike_2, spike_moving_average=sma_value) > expected_variance

    # Apply another spike, the variance should adjust based on the new mean
    spike_3 = 5
    sma_value = sma.apply(spike=spike_3)  # Mean should increase
    variance_before = vma.tracked_value()
    assert vma.apply(
        spike=spike_3, spike_moving_average=sma_value) > variance_before
