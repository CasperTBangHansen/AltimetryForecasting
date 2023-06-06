import torch
import torch.nn as nn
from typing import Tuple, Callable, Dict, Any

__all__ = ["Activation", "ConvLSTMCell", "ConvLSTM", "Seq2Seq"]

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

class ConvLSTM(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            padding: Tuple[int, int],
            activation: Activation,
            frame_size: Tuple[int, int],
            device: torch.device
        ):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.device = device

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,  kernel_size, padding, activation, frame_size, True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X is a frame sequence (batch_size, seq_len, num_channels, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, height, width, device = self.device)
        
        # Initialize Hidden State
        hidden_state = torch.zeros(batch_size, self.out_channels, height, width, device = self.device)

        # Initialize Cell Input
        cell_input = torch.zeros(batch_size, self.out_channels, height, width, device = self.device)

        # Unroll over time steps
        for time_step in range(seq_len):
            hidden_state, cell_input = self.convLSTMcell(X[:, :, time_step], hidden_state, cell_input)
            output[:, :, time_step] = hidden_state
        return output

class Seq2Seq(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_kernels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int], 
        activation: Activation,
        frame_size: Tuple[int, int],
        num_layers: int,
        device: torch.device
    ):
        super(Seq2Seq, self).__init__()
        if isinstance(activation, tuple):
            activation = activation[0]
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation,
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.device = device
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels = num_channels,
                out_channels = num_kernels,
                kernel_size = kernel_size,
                padding = padding, 
                activation = activation,
                frame_size = frame_size,
                device = device
            )
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features = num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers + 1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels = num_kernels,
                    out_channels = num_kernels,
                    kernel_size = kernel_size,
                    padding = padding, 
                    activation = activation,
                    frame_size = frame_size,
                    device = device
                )
            )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
            ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def get_kwargs(self) -> Dict[str, Any]:
        """ Returns the input arguments given to this class"""
        return {
            'num_channels': self.num_channels,
            'num_kernels': self.num_kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': self.activation,
            'frame_size': self.frame_size,
            'num_layers': self.num_layers,
            'device': self.device
        }       
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Forward propagation through all the layers
        output = self.sequential(X)
        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        return output