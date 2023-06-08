import torch
import torch.nn as nn
from typing import Tuple, Callable, Dict, Any, Optional, Sequence
from ConvLSTM import ConvLSTM

__all__ = ["Encoder", "Decoder", "Seq2SeqEncoderDecoder"]

Activation = Callable[[torch.Tensor], torch.Tensor]


    
class Encoder(nn.Module):

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
        super(Encoder, self).__init__()
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

        if isinstance(self.kernel_size, Sequence):
            kernel_size = self.kernel_size[0]

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
            if isinstance(self.kernel_size, Sequence):
                kernel_size = self.kernel_size[l - 1]

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
            
    def forward(self, X: torch.Tensor, layer_state: Dict[str, Dict[str, torch.Tensor]] | None = None) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
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
        return X, layer_state

class Decoder(nn.Module):

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
        super(Decoder, self).__init__()
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
        self.n_sequences = n_sequences

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
    
    def forward(self, X: torch.Tensor, layer_state: Dict[str, Dict[str, torch.Tensor]] | None = None) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        if layer_state is None:
            layer_state = {}
        return self.conv(X[:, :, -1]), layer_state

    
class Seq2SeqEncoderDecoder(nn.Module):

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
        super(Seq2SeqEncoderDecoder, self).__init__()
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
        self.n_sequences = n_sequences
        self.encoder = Encoder(
            num_channels,
            num_kernels,
            kernel_size,
            padding, 
            activation,
            frame_size,
            num_layers,
            device,
            n_sequences
        )
        self.decoder = Decoder(
            num_channels,
            num_kernels,
            kernel_size,
            padding, 
            activation,
            frame_size,
            num_layers,
            device,
            n_sequences
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

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()
        encoder_state = None
        decoder_state = None
        if y is not None:
            y = y.unsqueeze(1)
        
        # Initialize output
        output = torch.zeros(batch_size, self.num_channels, self.n_sequences, height, width, device = self.device)
        
        # Forward propagation through all the layers and keep state
        encoding, encoder_state = self.encoder(X, encoder_state)
        output[: ,:, 0], decoder_state = self.decoder(encoding, decoder_state)
        
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
                
                encoding, encoder_state = self.encoder(X, encoder_state)
                output[:, :, seq], decoder_state = self.decoder(encoding, decoder_state)
        return output