import torch
import torch.nn as nn
from typing import Protocol, Type, Dict, Any, Tuple, Mapping
from pathlib import Path
from DatasetOperations import Loss, DatasetParameters
import sys
if sys.version_info >= (3,11):
    from typing import Self
else:
    Self = None

class SaveModel(Protocol):
    """Protocol for a model to be saved"""

    def get_kwargs(self) -> Dict[str, Any]:
        """Saves the input arguments of the class"""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dict of the model"""
        ...
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any:
        """Sets the statedict of the model"""
        ...
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward function for pytorch nn.Module"""
        ...
    
    def __call__(self, *args: Any) -> Any:
        """Forward function for pytorch nn.Module"""
        ...
    
    def train(self, mode: bool = True) -> Self:
        """Train the model"""
        ...
    
    def eval(self) -> Self:
        """Evaluate the model"""
        ...

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

def load_checkpoint(path: Path, model_type: Type[nn.Module]) -> Tuple[nn.Module, torch.optim.Optimizer, Loss, DatasetParameters]:
    """Loads the model, optimizer and loss from the path"""
    checkpoint = torch.load(path)
    
    # Model
    model = model_type(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer
    optimizer = checkpoint['optimizer_type'](model.parameters(), **checkpoint['optimizer_kwargs'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Loss
    loss = checkpoint['loss']
    
    # Dataset parameters
    dataset_parameters = DatasetParameters(**checkpoint['dataset_parameters'])
    return model, optimizer, loss, dataset_parameters

