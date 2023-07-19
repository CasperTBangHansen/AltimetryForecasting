import torch
import torch.nn as nn
from typing import Dict, Any, Sequence
from ..AttentionConvLSTM.EncoderDecoder import Encoder, Decoder
from AltimeterAutoencoder.src import autoencoder
import random

__all__ = ["Seq2SeqAttention", "InputModel", "OutputModel"]

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

    def forward(
        self,
        featurespace_encoder: autoencoder.Encoder,
        featurespace_decoder: autoencoder.Decoder,
        X: torch.Tensor,
        y: torch.Tensor,
        pred_seq_len: int,
        teacher_force_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]
            y = [batch_size, seq_len, height, width]
            pred_seq_len = number of output sequences
            teacher_force_ratio = percent chance to use teacher forcing [0, 1]

        Output:
            output = [batch_size, seq_len, height, width]
        """
        # Set autoencoder to not require gradients
        featurespace_encoder.eval()
        featurespace_decoder.eval()
        for param in featurespace_encoder.parameters():
            param.requires_grad = False
        for param in featurespace_decoder.parameters():
            param.requires_grad = False
        
        # Convert input to featurespace
        feature_space = featurespace_encoder(X.transpose(1, 2)).transpose(1, 2)
        
        # Run through input network and the encoder
        x_encoder_in = self.input_network(feature_space)        
        encoder_output, hidden, cell = self.encoder(x_encoder_in)
        
        # Get the dimensions
        batch_size, channels, seq_len, height, width = X.size()
        outputs = torch.zeros(batch_size, channels, pred_seq_len, height, width, device = hidden.device)
        
        # First input
        input_decoder = x_encoder_in[:, :, -1]
        
        # Run through the decoder
        for t in range(pred_seq_len):
            output, hidden, cell = self.decoder(input_decoder, encoder_output, hidden, cell)
            decoded_featurespace = self.output_network(output)
            output = featurespace_decoder(decoded_featurespace.unsqueeze(1))
            outputs[:, :, t] = output.squeeze(1)

            if random.random() < teacher_force_ratio:
                feature_space = featurespace_encoder(
                    y[:, t].unsqueeze(1).unsqueeze(1).transpose(1, 2)
                ).transpose(1, 2)
            else:
                feature_space = featurespace_encoder(output.transpose(1, 2)).transpose(1, 2)

            input_decoder = self.input_network(feature_space).squeeze(2)
        return outputs.squeeze(1)
    
    def get_latent_space(
        self,
        featurespace_encoder: autoencoder.Encoder,
        featurespace_decoder: autoencoder.Decoder,
        X: torch.Tensor,
        pred_seq_len: int,
    ) -> torch.Tensor:
        """
        Input:
            X = [batch_size, in_channels, seq_len, height, width]
            y = [batch_size, seq_len, height, width]
            pred_seq_len = number of output sequences
            teacher_force_ratio = percent chance to use teacher forcing [0, 1]

        Output:
            latent_space = [batch_size, latent_space, seq_len, height, width]
            output = [batch_size, seq_len, height, width]
        """
        # Set autoencoder to not require gradients
        featurespace_encoder.eval()
        featurespace_decoder.eval()
        for param in featurespace_encoder.parameters():
            param.requires_grad = False
        for param in featurespace_decoder.parameters():
            param.requires_grad = False
        
        # Convert input to featurespace
        feature_space = featurespace_encoder(X.transpose(1, 2)).transpose(1, 2)

        # Run through input network and the encoder
        x_encoder_in = self.input_network(feature_space)        
        encoder_output, hidden, cell = self.encoder(x_encoder_in)
        
        # Get the dimensions
        batch_size, channels, seq_len, height, width = X.size()
        outputs = torch.zeros(batch_size, channels, pred_seq_len, height, width, device = hidden.device)
        latent_space = torch.zeros(batch_size, featurespace_encoder.feature_dimension, pred_seq_len, *self.encoder.frame_size, device = hidden.device)
    
        # First input
        input_decoder = x_encoder_in[:, :, -1]
        
        # Run through the decoder
        for t in range(pred_seq_len):
            output, hidden, cell = self.decoder(input_decoder, encoder_output, hidden, cell)
            decoded_featurespace = self.output_network(output)
            
            # decoded_featurespace = feature_space_[:, :, t]
            latent_space[:, :, t] = decoded_featurespace
            output = featurespace_decoder(decoded_featurespace.unsqueeze(1))
            outputs[:, :, t] = output.squeeze(1)
            feature_space = featurespace_encoder(output.transpose(1, 2)).transpose(1, 2)

            input_decoder = self.input_network(feature_space).squeeze(2)
        return latent_space, outputs.squeeze(1)
        