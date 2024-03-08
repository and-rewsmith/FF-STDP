from typing import Tuple
import pytest
import torch

from model.src.util import InhibitoryPlasticityTrace, SpikeMovingAverage, DoubleExponentialFilter, VarianceMovingAverage

# TODO: consider testing with real tau constants and dt values

device = torch.device("cpu")


def test_temporal_filter() -> None:
    tf = DoubleExponentialFilter(device=device, tau_rise=1, tau_fall=1)

    # Initial apply since temporal filter is second order
    tf.apply(value=torch.Tensor([1]), dt=1)

    # Apply a single error with a small dt should result in small change
    assert tf.apply(value=torch.Tensor([1]), dt=1).item(
    ) == pytest.approx(1.097208857536316)

    # Apply a zero error should result in decay of the state
    assert tf.apply(value=torch.Tensor([0]), dt=1).item(
    ) == pytest.approx(0.7217329740524292)


def test_spike_moving_average() -> None:
    sma = SpikeMovingAverage(device=device, batch_size=1, tau_mean=1, data_size=1)

    # Apply a single spike
    assert sma.apply(spike=torch.Tensor([1]), dt=1).item() == 0.6321205496788025

    # Apply another spike, the average should increase
    assert sma.apply(spike=torch.Tensor([2]), dt=1).item() == 1.496785283088684

    # After some time with no spikes, the average should decay
    assert sma.apply(spike=torch.Tensor([0]), dt=1).item() == 0.5506365299224854


def test_variance_moving_average() -> None:
    tau_mean = 10
    tau_var = 10
    sma = SpikeMovingAverage(device=device, batch_size=1, data_size=1, tau_mean=tau_mean)
    vma = VarianceMovingAverage(device=device, tau_var=tau_var)

    # Apply spikes to the moving average
    spike = torch.Tensor([2])
    for _ in range(20):
        # NOTE: The spike moving average is always updated first
        sma.apply(spike=spike, dt=1)
        vma.apply(spike=spike, spike_moving_average=sma.tracked_value(), dt=1)
    assert sma.tracked_value().item() > 1

    sma_value = sma.apply(spike=spike, dt=1)
    expected_variance = 0.38893842697143555
    assert vma.apply(
        spike=spike, spike_moving_average=sma_value, dt=1).item() == pytest.approx(expected_variance)

    # After some time with no spikes, the variance should increase
    spike_2 = torch.Tensor([0])
    sma_value = sma.apply(spike=spike_2, dt=1)  # Mean should decay
    assert vma.apply(
        spike=spike_2, spike_moving_average=sma_value, dt=1).item() > expected_variance

    # Apply another spike, the variance should adjust based on the new mean
    spike_3 = torch.Tensor([5])
    sma_value = sma.apply(spike=spike_3, dt=1)  # Mean should increase
    variance_before = vma.tracked_value().item()
    assert vma.apply(
        spike=spike_3, spike_moving_average=sma_value, dt=1).item() > variance_before


def test_inhibitory_plasticity_trace() -> None:
    trace_shape: Tuple[int, int] = (1, 1)
    ipt = InhibitoryPlasticityTrace(device=device, trace_shape=trace_shape)

    # Apply a single spike
    assert ipt.apply(spike=torch.Tensor([[1]]), dt=0.1).item() == 1

    # Apply another spike, the trace should increase
    assert ipt.apply(spike=torch.Tensor([[2]]), dt=0.1).item() > 2

    # After much time with no spikes, the trace should decay
    assert ipt.apply(spike=torch.Tensor([[0]]), dt=0.1).item() > 0
