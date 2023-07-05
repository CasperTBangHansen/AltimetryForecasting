import torch
import torch.nn as nn
from typing import Tuple, Callable

__all__ = ["Activation", "ConvLSTMCell"]

Activation = Callable[[torch.Tensor], torch.Tensor]

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        activation: Activation,
        frame_size: Tuple[int, int],
        bias: bool
    ):
        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.bias = bias
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels = in_channels + out_channels, 
            out_channels = 4 * out_channels, 
            kernel_size = kernel_size, 
            padding = padding,
            bias = bias
        )

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X: torch.Tensor, hidden_prev: torch.Tensor, cell_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, hidden_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * cell_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * cell_prev )

        # Current Cell output
        cell_output = forget_gate * cell_prev + input_gate * self.activation(C_conv)
        output_gate = torch.sigmoid(o_conv + self.W_co * cell_output )

        # Current Hidden State
        hidden_state = output_gate * self.activation(cell_output)

        return hidden_state, cell_output