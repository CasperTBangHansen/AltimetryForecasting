import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, Dict, Any
from ConvLSTM import ConvLSTMCell, Activation
from einops.layers.torch import Reduce
import random
__all__ = ["Seq2SeqED", "Encoder", "Decoder"]


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
        self.out_channels = out_channels
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size, True)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        hidden_state = torch.zeros(batch_size, self.out_channels, height, width, device = X.device)
        cell_input = torch.zeros(batch_size, self.out_channels, height, width, device = X.device)
        
        # Unroll over time steps
        for time_step in range(seq_len):
            hidden_state, cell_input = self.convLSTMcell(X[:, :, time_step], hidden_state, cell_input)
        return hidden_state, cell_input

class Decoder(nn.Module):

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
        self.out_channels = out_channels
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size, True)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor, cell_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_state, cell_input = self.convLSTMcell(x, hidden_state, cell_input)
        return hidden_state, hidden_state, cell_input

class Seq2SeqED(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = 1
        self.num_kernels = [10, 1]
        self.conv_kernel_size = [(3, 3), (3, 3), (3,3)]
        self.activation = nn.Tanh()
        self.frame_size = (128, 360)
        
        self.encoder = Encoder(
            self.input_channels,
            self.num_kernels[0],
            self.conv_kernel_size[0],
            ((self.conv_kernel_size[0][0] - 1)//2, (self.conv_kernel_size[0][1] - 1)//2),
            self.activation,
            self.frame_size,
        )
        self.decoder = Decoder(
            self.num_kernels[0],
            self.num_kernels[1],
            self.conv_kernel_size[1],
            ((self.conv_kernel_size[1][0] - 1)//2, (self.conv_kernel_size[1][1] - 1)//2),
            self.activation,
            self.frame_size,
        )
        self.max_pooling_layer = Reduce('b c h w -> b 1 h w', 'max')
        
        self.input_network = lambda x: x
        self.output_network = nn.Sequential(
            OrderedDict([
              ('out_linear_1', nn.Linear(10, 5)),
              ('out_linear_2', nn.Linear(5, 1)),
            ])
        )
        
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {}

    def forward(self, X: torch.Tensor, y: torch.Tensor, seq_len: int, teacher_force_ratio: float = 0.5) -> torch.Tensor:
        X = self.input_network(X)        
        hidden, cell = self.encoder(X)
        
        # Get the dimensions
        batch_size, channels, height, width = hidden.size()
        outputs = torch.zeros(batch_size, self.num_kernels[-1], seq_len, height, width, device = hidden.device)
        
        input_decoder = torch.zeros(batch_size, self.num_kernels[-1], height, width, device = hidden.device)
        # First input
        for t in range(seq_len):
            output, hidden, cell = self.decoder(input_decoder, hidden, cell)
            output = self.output_network(
                output.transpose(1, 3)
            ).transpose(3, 1)
            outputs[:, :, t] = output
            input_decoder = y[:, t].unsqueeze(1) if random.random() < teacher_force_ratio else output
        
        return outputs.squeeze(1)