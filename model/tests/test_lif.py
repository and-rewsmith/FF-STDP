import torch

from model.src.lif import LIF


def test_lif_spikes_for_membrane_above_0():
    lif = LIF(beta=0.9, threshold=1.0)

    # make sure no spike at first
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 0).all()
    assert (lif.mem > 1).all()

    # make sure spike at second based on previous current
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 1).all()
    # make sure membrane potential is below 1 indicating a decay since the last timestep
    assert (lif.mem < 1).all()


def test_lif_spikes_for_membrane_below_0():
    lif = LIF(beta=0.9, threshold=1.0)

    # make sure no spike at first
    input = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 0).all()
    assert (lif.mem > 0).all()

    # make sure no spike at second based on previous current
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 0).all()
    assert (lif.mem > 0).all()


def test_lif_spikes_then_decays_and_does_not_spike():
    lif = LIF(beta=0.9, threshold=1.0)

    # make sure no spike at first
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 0).all()
    assert (lif.mem > 1).all()

    # make sure spike at second based on previous current
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 1).all()
    assert (lif.mem < 1).all()

    # make sure no spike at third based on previous current
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 0).all()
    assert (lif.mem < 1).all()
