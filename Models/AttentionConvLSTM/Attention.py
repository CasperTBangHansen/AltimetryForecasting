import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
__all__ = ["Attention"]

class Attention(nn.Module):

    def __init__(self, encoding_hidden_dim: int, decoding_hidden_dim: int):
        super().__init__()
        # 2 * encoding for forward and backwards LSTMS
        # decoding for the hidden state
        self.encoding_hidden_dim = encoding_hidden_dim
        self.decoding_hidden_dim = decoding_hidden_dim
        self.attention = nn.Linear((encoding_hidden_dim * 2) + decoding_hidden_dim, decoding_hidden_dim)
        self.weights = nn.Linear(decoding_hidden_dim, 1, bias = False)

    def forward(self, hidden_state: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Input:
            hidden_state = [batch_size, decoding_hidden_dim, height, width]
            encoder_outputs = [batch_size, encoding_hidden_dim * 2, seq_len, height, width]

        Output:
            output = [batch_size, out_channels * 2, seq_len, height, width]
            hidden_state = [batch_size, out_channels, height, width]
            cell_input = [batch_size, out_channels, height, width]
        """
        
        # Repeat hidden state one for each encoding output
        hidden_state = hidden_state.unsqueeze(2).repeat(1, 1, encoder_outputs.size(2), 1, 1)
        
        # Compute energy
        # Transpose so the channels are in the last layer so the linear layers
        # and the last layer have the same dimension
        stacked_tensors = torch.cat((hidden_state, encoder_outputs), dim = 1)
        inner_energy = self.attention(stacked_tensors.transpose(1, -1))
        energy = torch.tanh(inner_energy)
                
        # Compute attention
        # Transpose back to the original size
        attention = self.weights(energy).transpose(-1, 1).squeeze(1)
                
        # Return weights of each encoder output
        return F.softmax(attention, dim=1)

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            'encoding_hidden_dim': self.encoding_hidden_dim,
            'decoding_hidden_dim': self.decoding_hidden_dim
        }