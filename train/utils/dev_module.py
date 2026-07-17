"""Minimal DevModule replacement for torch-enhanced compatibility."""

import torch
import torch.nn as nn


class DevModule(nn.Module):
    """
    Base class that extends nn.Module with a device property.
    Replaces `torchenhanced.DevModule`.
    """

    @property
    def device(self) -> str:
        """Return the device string of the first parameter, or 'cpu' if no parameters."""
        params = list(self.parameters())
        if params:
            return str(params[0].device)
        buffers = list(self.buffers())
        if buffers:
            return str(buffers[0].device)
        return 'cpu'
