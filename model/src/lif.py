from typing import Any, Optional

import torch
import torch.nn as nn


class LIF(nn.Module):
    def __init__(self, beta: float, threshold: float = 1.0):
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
        # Initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_op = self.SpikeOperator.apply
        self.mem: Optional[torch.Tensor] = None
        self.prereset_mem: Optional[torch.Tensor] = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.mem == None:  # noqa
            self.mem = torch.zeros_like(current)
            self.prereset_mem = torch.zeros_like(current)

        # Update membrane potential: decay and add current
        self.mem = self.beta * self.mem + current
        self.prereset_mem = self.mem.clone()

        # Spike if membrane potential exceeds threshold
        spk: torch.Tensor = self.spike_op(self.mem, self.threshold)

        # Reset the membrane potential if spiked
        reset = spk * self.threshold
        self.mem -= reset  # type: ignore [operator]

        return spk

    class SpikeOperator(torch.autograd.Function):

        @staticmethod
        def forward(ctx: Any, mem: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
            input = mem - threshold
            spk = torch.zeros_like(input)
            spk[input > 0] = 1.0
            return spk
