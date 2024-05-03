from typing import Optional

import torch
import torch.nn as nn


class SpikeOperator():
    @staticmethod
    def forward(mem: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        spk = torch.where(mem > threshold, torch.as_tensor(1.0, device=mem.device),
                          torch.as_tensor(0.0, device=mem.device))
        return spk


class LIF(nn.Module):
    def __init__(self, beta: float, threshold_scale: float, threshold_decay: float, threshold: float = 1.0):
        """
        NOTE: Initialization of mem to zeros at initial forward pass will cause
        transient behavior

        NOTE: The membrane potential is not resting at a negative value, as it
        is in Zenke's work. This implementation may need to change.

        NOTE: The ordering of the following operations within a timestep is
        important. So we need to pick the correct ordering. The current
        implementation follows Zenke's work, HOWEVER, the membrane potential
        used in the learning rule is separate from this implementation, and may
        either use the `mem` or `prereset_mem`.

        Order of operations:
        1. Update membrane potential
        2. Spike if membrane potential exceeds threshold
        3. Reset the membrane potential if spiked
        """
        super(LIF, self).__init__()
        assert 0 <= beta <= 1, "beta must be in the range [0, 1]"
        assert threshold_scale >= 1, "threshold_scale must be greater than or equal to 1"
        assert 0 <= threshold_decay <= 1, "threshold_decay must be in the range [0, 1]"

        self.beta = beta
        self.threshold = threshold
        self.threshold_scale = threshold_scale
        self.threshold_decay = threshold_decay
        self.spike_op = SpikeOperator.forward

        self.mem: torch.Tensor
        self.prereset_mem: torch.Tensor
        self.adaptive_threshold: torch.Tensor

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'mem'):
            self.mem = torch.zeros_like(current)
            self.prereset_mem = torch.zeros_like(current)
            self.adaptive_threshold = torch.full_like(current, self.threshold)

        # Update membrane potential: decay and add current
        self.mem = self.beta * self.mem + current
        self.prereset_mem = self.mem.clone()

        # Spike if membrane potential exceeds adaptive threshold
        spk: torch.Tensor = self.spike_op(self.mem, self.adaptive_threshold)

        # Reset the membrane potential if spiked
        reset = spk * self.adaptive_threshold
        self.mem -= reset

        # Update adaptive threshold
        self.adaptive_threshold = torch.where(spk.bool(), self.adaptive_threshold * self.threshold_scale,
                                              self.adaptive_threshold * self.threshold_decay)

        return spk
