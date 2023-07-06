import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .ConvLSTM import ConvLSTMCell, Activation, ConvLSTMLayer
from .Attention import Attention
__all__ = ["Encoder", "Decoder"]

class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        activation: Activation,
        frame_size: Tuple[int, int],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.convLSTMcell = ConvLSTMLayer(in_channels, out_channels, kernel_size, padding, activation, frame_size, True)
        self.fc_hidden = nn.Linear(out_channels*2, out_channels)
        self.fc_cell = nn.Linear(out_channels*2, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]

        Output:
            output = [batch_size, out_channels * 2, seq_len, height, width]
            hidden_state = [batch_size, out_channels, height, width]
            cell_input = [batch_size, out_channels, height, width]
        """
        
        output, hidden_state, cell_input = self.convLSTMcell(x)

        hidden_state = self.fc_hidden(hidden_state.transpose(1, 3)).transpose(3, 1)
        cell_input = self.fc_cell(cell_input.transpose(1, 3)).transpose(3, 1)
        return output, hidden_state, cell_input
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': self.activation,
            'frame_size': self.frame_size,
        }


class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        activation: Activation,
        frame_size: Tuple[int, int],
        attention: Attention
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.attention = attention
        self.convLSTMcell = ConvLSTMCell(in_channels + 2 * hidden_channels, out_channels, kernel_size, padding, activation, frame_size, True)

    @property
    def resulting_channels(self) -> int:
        return self.out_channels + self.in_channels + 2 * self.hidden_channels
        
    def forward(
        self,
        X: torch.Tensor,
        encoder_outputs: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            X = [batch_size, in_channels, height, width]
            encoder_outputs = [batch_size, in_channels * 2, seq_len, height, width]
            hidden_state = [batch_size, in_channels, height, width]
            cell_input = [batch_size, in_channels, height, width]

        Output:
            skip_connecting_and_weigths = [batch_size, out_channels + in_channels + 2 * hidden_channels, height, width]
            hidden_state = [batch_size, in_channels, height, width]
            cell_input = [batch_size, in_channels, height, width]
        """

        weights = self.attention(hidden_state, encoder_outputs)
        context_vector = torch.einsum("b t h w, b c t h w -> b c h w", weights, encoder_outputs)
        lstm_input = torch.cat((context_vector, X), dim=1)
        output, hidden_state, cell_input = self.convLSTMcell(lstm_input, hidden_state, cell_input)
        
        skip_connecting_and_weigths = torch.cat((output, context_vector, X), dim = 1)

        return skip_connecting_and_weigths, hidden_state, cell_input

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': self.activation,
            'frame_size': self.frame_size,
            'attention': self.attention.get_kwargs()
        }