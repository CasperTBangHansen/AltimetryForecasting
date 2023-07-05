from typing import Type
import torch.nn as nn
import torch
from .training_loop import loss_function

def create_masked_loss_function(loss_module: Type[nn.modules.loss._Loss]) -> loss_function:
    """Constructs a masked loss function based on the loss module"""
    _loss_function = loss_module()
    def masked_loss_function(output: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """
        Maskes the output and target based on the mask.
        The mask is True where the values should be ignored.
        """
        if len(output.shape) == 4:
            total_loss = torch.tensor(0, device = output.device, dtype = output.dtype)
            # Sum loss for each day
            for day_idx in range(output.size(1)):
                mask = ~masked[:, day_idx]
                loss = _loss_function(output[:, day_idx][mask], target[:, day_idx][mask])
                total_loss += loss
            # Batch norm
            total_loss /= output.size(0)
        else:
            total_loss = _loss_function(output[masked], target[masked])
        return total_loss
    return masked_loss_function