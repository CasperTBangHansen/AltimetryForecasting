import torch
from typing import Type, Tuple
from pathlib import Path
from . import Attention, InputModel, OutputModel, Encoder, Decoder, Seq2SeqAttention
from ..Shared import SaveModel, Loss, DatasetParameters

def save_checkpoint(path: Path, model: SaveModel, optimizer: torch.optim.Optimizer, loss: Loss, dataset_parameters: DatasetParameters):
    """Saves the model and the parameters needed to load the checkpoint again"""
    torch.save(
        {
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_type': type(optimizer),
            'optimizer_kwargs': optimizer.defaults,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'dataset_parameters': dataset_parameters.__dict__
        },
        path
    )

def load_checkpoint(path: Path, model_type: Type[Seq2SeqAttention], device: torch.device) -> Tuple[Seq2SeqAttention, torch.optim.Optimizer, Loss, DatasetParameters]:
    """Loads the model, optimizer and loss from the path"""
    checkpoint = torch.load(path, map_location=device)
    
    # Model
    checkpoint['device'] = device
    input_model = InputModel(**checkpoint['model_kwargs'].pop('input_network'))
    encoder = Encoder(**checkpoint['model_kwargs'].pop('encoder'))
    attention = Attention(**checkpoint['model_kwargs']['decoder'].pop('attention'))
    decoder = Decoder(attention=attention, **checkpoint['model_kwargs'].pop('decoder'))
    output_model = OutputModel(**checkpoint['model_kwargs'].pop('output_network'))
    
    model = model_type(
        input_network = input_model,
        encoder = encoder,
        decoder = decoder,
        output_network = output_model,
        **checkpoint['model_kwargs']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Optimizer
    optimizer = checkpoint['optimizer_type'](model.parameters(), **checkpoint['optimizer_kwargs'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Loss
    loss = checkpoint['loss']
    
    # Dataset parameters
    dataset_parameters = DatasetParameters(**checkpoint['dataset_parameters'])
    return model, optimizer, loss, dataset_parameters