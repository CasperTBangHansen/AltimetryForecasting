from typing import Type
import torch.nn as nn
import numpy as np
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
            total_loss /= output.size(1)
        else:
            total_loss = _loss_function(output[masked], target[masked])
        return total_loss
    return masked_loss_function

def create_masked_loss_function_diff(loss_module: Type[nn.modules.loss._Loss]) -> loss_function:
    """Constructs a masked loss function based on the loss module"""
    _loss_function = loss_module()
    def masked_loss_function(output: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """
        Maskes the output and target based on the mask.
        The mask is True where the values should be ignored.
        """
        mse_losses = np.zeros(output.size(1))
        grad_losses = np.zeros(output.size(1) - 1)
        mse_loss: torch.Tensor = torch.tensor(0, device = output.device, dtype = output.dtype)
        grad_loss: torch.Tensor = torch.tensor(0, device = output.device, dtype = output.dtype)
        # Sum loss for each day
        for day_idx in range(output.size(1)):
            mask = ~masked[:, day_idx]
            loss = _loss_function(output[:, day_idx][mask], target[:, day_idx][mask]) * np.log(day_idx + 2)
            mse_losses[day_idx] = loss.detach().item()
            mse_loss += loss
        
        # Gradient in time
        for day_idx in range(1, output.size(1)):
            mask = (~masked[:, day_idx - 1] | ~masked[:, day_idx])
            output_grad = output[:, day_idx - 1] - output[:, day_idx]
            target_grad = target[:, day_idx - 1] - target[:, day_idx]
            loss = _loss_function(output_grad[mask], target_grad[mask]) * np.log(day_idx + 1)
            grad_losses[day_idx - 1] = loss.detach().item()
            grad_loss += loss
        
        # Batch norm
        mse_loss /= output.size(1)
        grad_loss /= output.size(1)
        return mse_loss + grad_loss, mse_losses, grad_losses
    return masked_loss_function