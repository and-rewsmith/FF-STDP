import torch

from model.src.lif import LIF


def test_membrane_behaves_for_lif_spikes_for_membrane_above_0() -> None:
    lif = LIF(beta=0.9, threshold=1.0)

    # make sure spikes at first forward
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 1).all()  # type: ignore

    # make sure that the spike resets the membrane in the same timestep
    assert (lif.mem < 2).all()  # type: ignore
    assert (lif.prereset_mem == 2).all()  # type: ignore

    # make sure no spike at second based on previous current
    prev_lif_mem = lif.mem.clone()
    prev_lif_prereset_mem = lif.prereset_mem.clone()
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 0).all()  # type: ignore

    # confirm membrane decays
    assert (lif.mem < prev_lif_mem).all()  # type: ignore
    assert (lif.prereset_mem < prev_lif_prereset_mem).all()  # type: ignore

    # confirm membrane prereset mem is the same as the membrane
    assert (lif.prereset_mem == lif.mem).all()  # type: ignore


def test_lif_no_spikes_for_membrane_below_0() -> None:
    lif = LIF(beta=0.9, threshold=1.0)

    # make sure no spike at first
    input = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 0).all()  # type: ignore
    assert (lif.mem > 0).all()  # type: ignore

    # make sure no spike at second based on previous current
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 0).all()  # type: ignore
    assert (lif.mem > 0).all()  # type: ignore


def test_lif_spikes_repeatedly() -> None:
    lif = LIF(beta=0.9, threshold=1.0)

    # make sure spikes at first forward
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 1).all()  # type: ignore
    assert (lif.mem == 1).all()  # type: ignore

    # make sure that spikes at second forward
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 1).all()  # type: ignore
    assert (lif.mem > 1.5).all()  # type: ignore
