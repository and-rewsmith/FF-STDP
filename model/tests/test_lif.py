import torch

from model.src.lif import LIF


def test_membrane_behaves_for_lif_spikes_for_membrane_above_0() -> None:
    lif = LIF(beta=0.9, threshold=1.0, threshold_scale=1, threshold_decay=1)

    # make sure spikes at first forward
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = lif(input)
    assert spk.shape == input.shape
    assert (spk == 1).all()  # type: ignore

    # make sure that the spike resets the membrane in the same timestep
    assert (lif.mem < 2).all()  # type: ignore
    assert (lif.prereset_mem == 2).all()  # type: ignore

    # make sure no spike at second based on previous current
    prev_lif_mem = lif.mem.clone()  # type: ignore
    prev_lif_prereset_mem = lif.prereset_mem.clone()  # type: ignore
    input = torch.tensor([[0, 0, 0], [0, 0, 0]])
    spk = lif(input)
    assert (spk == 0).all()  # type: ignore

    # confirm membrane decays
    assert (lif.mem < prev_lif_mem).all()  # type: ignore
    assert (lif.prereset_mem < prev_lif_prereset_mem).all()  # type: ignore

    # confirm membrane prereset mem is the same as the membrane
    assert (lif.prereset_mem == lif.mem).all()  # type: ignore


def test_lif_no_spikes_for_membrane_below_0() -> None:
    lif = LIF(beta=0.9, threshold=1.0, threshold_scale=1, threshold_decay=1)

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
    lif = LIF(beta=0.9, threshold=1.0, threshold_scale=1, threshold_decay=1)

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


def test_alif_adaptive_threshold_behavior() -> None:
    alif = LIF(beta=0.9, threshold=1.0, threshold_scale=1.2, threshold_decay=0.95)

    # make sure spikes at first forward and adaptive threshold increases
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = alif(input)
    assert (spk == 1).all()  # type: ignore
    assert (alif.adaptive_threshold == 1.2).all()  # type: ignore

    # make sure no spike at second forward and adaptive threshold decays
    input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    spk = alif(input)
    assert (spk == 0).all()  # type: ignore
    assert (alif.adaptive_threshold == 1.14).all()  # type: ignore

    # make sure spikes at third forward and adaptive threshold increases again
    input = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    spk = alif(input)
    assert (spk == 1).all()  # type: ignore
    assert (alif.adaptive_threshold == 1.368).all()  # type: ignore


def test_alif_behavior_with_different_parameters() -> None:
    alif = LIF(beta=0.8, threshold=0.5, threshold_scale=1.1, threshold_decay=0.9)

    # make sure spikes at first forward
    input = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    spk = alif(input)
    assert (spk == 1).all()  # type: ignore
    assert (alif.mem == 0.5).all()  # type: ignore
    assert (alif.adaptive_threshold == 0.55).all()  # type: ignore

    # make sure spikes at second forward
    input = torch.tensor([[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]])
    spk = alif(input)
    assert (spk == 1).all()  # type: ignore
    assert torch.allclose(alif.mem, torch.tensor([[0.45], [0.45]]))
    assert torch.allclose(alif.adaptive_threshold, torch.tensor([[0.605], [0.605]]))
