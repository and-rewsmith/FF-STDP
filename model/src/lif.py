from typing import Any, Optional

import torch
import torch.nn as nn


class LIF(nn.Module):
    def __init__(self, beta: float, threshold: float = 1.0):
        """
        NOTE: Initialization of mem to zeros at initial forward pass will cause
        transient behavior
        """
        super(LIF, self).__init__()
        # Initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_op = self.SpikeOperator.apply
        self.mem: Optional[torch.Tensor] = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.mem == None:  # noqa
            self.mem = torch.zeros_like(current)

        spk: torch.Tensor = self.spike_op(self.mem, self.threshold)
        reset = spk * self.threshold
        self.mem = self.beta * self.mem + current - reset  # type: ignore [operator]
        return spk

    class SpikeOperator(torch.autograd.Function):

        @staticmethod
        def forward(ctx: Any, mem: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
            input = mem - threshold
            spk = torch.zeros_like(input)
            spk[input > 0] = 1.0
            return spk
