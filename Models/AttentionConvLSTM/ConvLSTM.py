import torch
import torch.nn as nn
from typing import Tuple, Callable

__all__ = ["Activation", "ConvLSTMCell", "ConvLSTMLayer"]

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

    def forward(self, X: torch.Tensor, hidden_prev: torch.Tensor, cell_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            X = [batch_size, in_channels, height, width]
            hidden_prev = [batch_size, out_channels, height, width]
            cell_prev = [batch_size, out_channels, height, width]

        Output:
            output = [batch_size, out_channels, seq_len, height, width]
            hidden_state = [batch_size, out_channels, height, width]
            cell_input = [batch_size, out_channels, height, width]
        """
        
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

        return hidden_state, hidden_state, cell_output

class ConvLSTMLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        activation: Activation,
        frame_size: Tuple[int, int],
        bidirectional = False
    ):
        super().__init__()
        self.out_channels = out_channels
        self.bidirectional = bidirectional
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size, True)

    def _forward_parse(self, X: torch.Tensor, forward_direction: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]

        Output:
            output = [batch_size, out_channels, seq_len, height, width]
            hidden_state = [batch_size, out_channels, height, width]
            cell_input = [batch_size, out_channels, height, width]
        """
        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, height, width, device = X.device)
        hidden_state = torch.zeros(batch_size, self.out_channels, height, width, device = X.device)
        cell_input = torch.zeros(batch_size, self.out_channels, height, width, device = X.device)
        
        # Unroll over time steps
        time_steps = range(seq_len)
        if not forward_direction:
            time_steps = reversed(time_steps)
        for time_step in time_steps:
            output[:, :, time_step], hidden_state, cell_input = self.convLSTMcell(X[:, :, time_step], hidden_state, cell_input)
        return output, hidden_state, cell_input
    
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]

        Output:
            (bidirectional = False)
                output = [batch_size, out_channels, seq_len, height, width]
                hidden_state = [batch_size, out_channels, height, width]
                cell_input = [batch_size, out_channels, height, width]

            (bidirectional = True)
                output = [batch_size, out_channels * 2, seq_len, height, width]
                hidden_state = [batch_size, out_channels * 2, height, width]
                cell_input = [batch_size, out_channels * 2, height, width]
        """
        if not self.bidirectional:
            return self._forward_parse(X, True)
        
        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels * 2, seq_len, height, width, device = X.device)
        hidden_state = torch.zeros(batch_size, self.out_channels * 2, height, width, device = X.device)
        cell_input = torch.zeros(batch_size, self.out_channels * 2, height, width, device = X.device)
        
        output[:, :self.out_channels], hidden_state[:, :self.out_channels], cell_input[:, :self.out_channels] = self._forward_parse(X, True)
        output[:, self.out_channels:], hidden_state[:, self.out_channels:], cell_input[:, self.out_channels:] = self._forward_parse(X, False)
        return output, hidden_state, cell_input