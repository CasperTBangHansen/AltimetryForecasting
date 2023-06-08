import torch
import torch.nn as nn
from typing import Tuple, Callable, Dict, Any, Optional, Sequence

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
    
    def predict_keep_state(self, X: torch.Tensor, hidden: torch.Tensor | None = None, cell: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # X is a frame sequence (batch_size, seq_len, num_channels, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, height, width, device = self.device)

        if hidden is None:
            # Initialize Hidden State
            hidden_state = torch.zeros(batch_size, self.out_channels, height, width, device = self.device)
        else:
            hidden_state = hidden

        if cell is None:
            # Initialize Cell Input
            cell_input = torch.zeros(batch_size, self.out_channels, height, width, device = self.device)
        else:
            cell_input = cell

        # Unroll over time steps
        for time_step in range(seq_len):
            hidden_state, cell_input = self.convLSTMcell(X[:, :, time_step], hidden_state, cell_input)
            output[:, :, time_step] = hidden_state
        
        return output, cell_input, hidden_state
    
class Seq2Seq(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_kernels: int | Sequence[int],
        kernel_size: Tuple[int, int] | Sequence[Tuple[int, int]],
        padding: Tuple[int, int], 
        activation: Activation,
        frame_size: Tuple[int, int],
        num_layers: int,
        device: torch.device,
        n_sequences: Optional[int] = 1
    ):
        super(Seq2Seq, self).__init__()
        if isinstance(activation, Sequence):
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
        self.n_sequences = n_sequences

        # Add First layer (Different in_channels than the rest)
        if isinstance(self.num_kernels, Sequence):
            num_kernels = self.num_kernels[0]
        self.sequential.add_module(
            "encoder_convlstm1", ConvLSTM(
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
            "encoder_batchnorm1", nn.BatchNorm3d(num_features = num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            if isinstance(self.num_kernels, Sequence):
                num_kernels = self.num_kernels[l - 1]

            self.sequential.add_module(
                f"encoder_convlstm{l}", ConvLSTM(
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
                f"encoder_batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
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
            'device': self.device,
            'n_sequences': self.n_sequences
        }       
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Forward propagation through all the layers
        output = self.sequential(X)
        
        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        return output
    
    def forward_state(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()
        layer_state = None
        if y is not None:
            y = y.unsqueeze(1)
        
        # Initialize output
        output = torch.zeros(batch_size, self.num_channels, self.n_sequences, height, width, device = self.device)
        
        # Forward propagation through all the layers and keep state
        output[: ,:, 0], layer_state = self._predict_keep_state(X, layer_state)
        
        if self.n_sequences > 1:
            for seq in range(1, self.n_sequences):
                try:
                    X[:, :, :-1] = X[:, :, 1:]
                except RuntimeError:
                    X[:, :, :-1] = X[:, :, 1:].clone()
                
                # y is None means no teacher forcing
                if y is None:
                    X[:, :, -1] = output[: ,:, seq - 1].detach()
                else:
                    # Use teacher forcing
                    X[:, :, -1] = y[: ,:, seq - 1]
                output[:, :, seq], layer_state = self._predict_keep_state(X, layer_state)
        return output
    
    def _predict_keep_state(self, X: torch.Tensor, layer_state: Dict[str, Dict[str, torch.Tensor]] | None = None) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        if layer_state is None:
            layer_state = {}
        
        for layer_name, layer in self.sequential.named_children():
            if isinstance(layer, ConvLSTM):
                if layer_name in layer_state:
                    X, hidden_state, cell_state = layer.predict_keep_state(
                        X,
                        layer_state[layer_name]['hidden_state'],
                        layer_state[layer_name]['cell_state']
                    )
                else:
                    X, hidden_state, cell_state = layer.predict_keep_state(X)

                layer_state[layer_name] = {}
                layer_state[layer_name]['hidden_state'] = hidden_state
                layer_state[layer_name]['cell_state'] = cell_state
            else:
                X = layer(X)
        output = self.conv(X[:, :, -1])
        return output, layer_state