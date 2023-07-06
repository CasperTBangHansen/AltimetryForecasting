import torch
import torch.nn as nn
from typing import Dict, Any, Sequence
from .EncoderDecoder import Encoder, Decoder
import random

__all__ = ["Seq2SeqAttentionZero", "Seq2SeqAttention", "InputModel", "OutputModel"]

class InputModel(nn.Module):
    def __init__(self, in_channels: int, output_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.input_network = nn.Linear(in_channels, output_channels)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]

        Output:
            output = [batch_size, output_channels, seq_len, height, width]
        """
        return self.input_network(X.transpose(1, -1)).transpose(-1, 1)
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {'in_channels': self.in_channels, 'output_channels': self.output_channels}

class OutputModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Sequence[int], output_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_network = nn.Sequential()
        if len(hidden_channels) > 0:
            self.output_network.add_module('input_layer', nn.Linear(in_channels, hidden_channels[0]))
            
            for i in range(1, len(hidden_channels)):
                self.output_network.add_module(f'hidden_{i}', nn.Linear(hidden_channels[i - 1], hidden_channels[i]))
            self.output_network.add_module('output_layer', nn.Linear(hidden_channels[-1], output_channels))
        else:
            self.output_network.add_module('layer', nn.Linear(in_channels, output_channels))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input:
            X = [batch_size, in_channels, height, width]

        Output:
            output = [batch_size, output_channels, height, width]
        """
        return self.output_network(X.transpose(1, -1)).transpose(-1, 1)
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'output_channels': self.output_channels
        }
    
class Seq2SeqAttention(nn.Module):
    def __init__(
        self,
        input_network: InputModel,
        encoder: Encoder,
        decoder: Decoder,
        output_network: OutputModel
    ):
        super().__init__()
        self.input_network = input_network
        self.encoder = encoder
        self.decoder = decoder
        self.output_network = output_network
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {
            'input_network': self.input_network.get_kwargs(),
            'encoder': self.encoder.get_kwargs(),
            'decoder': self.decoder.get_kwargs(),
            'output_network': self.output_network.get_kwargs()
        }

    def forward(self, X: torch.Tensor, y: torch.Tensor, pred_seq_len: int, teacher_force_ratio: float = 0.5) -> torch.Tensor:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]
            y = [batch_size, seq_len, height, width]
            pred_seq_len = number of output sequences
            teacher_force_ratio = percent chance to use teacher forcing [0, 1]

        Output:
            output = [batch_size, seq_len, height, width]
        """
        
        # Run through input network and the encoder
        x_encoder_in = self.input_network(X)        
        encoder_output, hidden, cell = self.encoder(x_encoder_in)
        
        # Get the dimensions
        batch_size, channels, height, width = hidden.size()
        outputs = torch.zeros(batch_size, 1, pred_seq_len, height, width, device = hidden.device)
        
        # First input
        input_decoder = x_encoder_in[:, :, -1]
        
        # Run through the decoder
        for t in range(pred_seq_len):
            output, hidden, cell = self.decoder(input_decoder, encoder_output, hidden, cell)
            output = self.output_network(output)
            outputs[:, :, t] = output
            
            if random.random() < teacher_force_ratio:
                input_decoder = self.input_network(y[:, t].unsqueeze(1))
            else:
                input_decoder = self.input_network(output)
        return outputs.squeeze(1)
    
class Seq2SeqAttentionZero(nn.Module):
    def __init__(
        self,
        input_network: InputModel,
        encoder: Encoder,
        decoder: Decoder,
        output_network: OutputModel
    ):
        super().__init__()
        self.input_network = input_network
        self.encoder = encoder
        self.decoder = decoder
        self.output_network = output_network
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {
            'input_network': self.input_network.get_kwargs(),
            'encoder': self.encoder.get_kwargs(),
            'decoder': self.decoder.get_kwargs(),
            'output_network': self.output_network.get_kwargs()
        }

    def forward(self, X: torch.Tensor, y: torch.Tensor, pred_seq_len: int, teacher_force_ratio: float = 0.5) -> torch.Tensor:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]
            y = [batch_size, seq_len, height, width]
            pred_seq_len = number of output sequences
            teacher_force_ratio = percent chance to use teacher forcing [0, 1]

        Output:
            output = [batch_size, seq_len, height, width]
        """
        
        # Run through input network and the encoder
        x_encoder_in = self.input_network(X)        
        encoder_output, hidden, cell = self.encoder(x_encoder_in)
        
        # Get the dimensions
        batch_size, channels, height, width = hidden.size()
        outputs = torch.zeros(batch_size, 1, pred_seq_len, height, width, device = hidden.device)
        
        # First input
        input_decoder = torch.zeros(batch_size, self.decoder.in_channels, height, width, device = hidden.device)
        
        # Run through the decoder
        for t in range(pred_seq_len):
            output, hidden, cell = self.decoder(input_decoder, encoder_output, hidden, cell)
            output = self.output_network(output)
            outputs[:, :, t] = output
            
            if random.random() < teacher_force_ratio:
                input_decoder = self.input_network(y[:, t].unsqueeze(1))
            else:
                input_decoder = self.input_network(output)
        return outputs.squeeze(1)