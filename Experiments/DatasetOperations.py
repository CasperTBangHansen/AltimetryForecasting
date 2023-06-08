import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import date, datetime
from torch.utils.data import Dataset

try:
    from ..Models import _types
except:
    from Models import _types

@dataclass
class DatasetParameters:
    """ Main hyperparameters to construct the dataset"""
    batch_size: int
    sequence_length: int
    sequence_steps: int
    prediction_steps: int
    fill_nan: float
    train_end: date
    validation_end: date
    n_predictions: Optional[int] = 1

@dataclass
class Loss:
    """Maps epoch id to training and validation loss"""
    epoch: int
    training: float
    validation: float
    time: Optional[datetime] = None
    learning_rate: Optional[float] = 0

class SLADataset(Dataset):
    """Constructs the SLA dataset"""
    def __init__(self, x: _types.float_like, t: _types.int_like, datasetparameters: DatasetParameters):
        # Convert to float32 (float64 is overkill)
        x_32 = x.astype(np.float32)
        
        self.mask = torch.tensor(np.isnan(x_32), dtype=torch.bool)
        self.x = torch.nan_to_num(torch.tensor(x_32, dtype=torch.float32), nan = datasetparameters.fill_nan)
        self.t = torch.tensor(t)
        self.sequence_length = datasetparameters.sequence_length
        self.sequence_steps = datasetparameters.sequence_steps
        self.prediction_steps = datasetparameters.prediction_steps
        self.n_predictions = datasetparameters.n_predictions
        self.fill_nan = datasetparameters.fill_nan
        self._len = len(self.x)
        self._sequence_size = (self.sequence_steps - 1) * (self.sequence_length - 1) + self.sequence_length

    def __len__(self) -> int:
        return self._len - self._sequence_size - self.prediction_steps * self.n_predictions

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Features
        features = self.x[idx:idx + self.sequence_steps*(self.sequence_length):self.sequence_steps]
        features_time = self.t[idx:idx + self.sequence_steps*(self.sequence_length):self.sequence_steps]
        
        # Target
        end_idxs = [idx + self._sequence_size + self.prediction_steps * i - 1 for i in range(1, self.n_predictions + 1)]
        target = self.x[end_idxs]
        mask = self.mask[end_idxs]
        pred_time = self.t[end_idxs]
        return features.unsqueeze(0), target, mask, features_time, pred_time

def stack_frames(dataset: _types.float_like, n_frames: int) -> _types.float_like:
    """
    Stacks each frame in the dataset into n_frames.
    Converts dataset of shape
        (num_frames, y_length, x_length)
    to
        (num_frames - n_frames, n_frames, y_length, x_length)
    """
    num_frames, y_length, x_length = dataset.shape

    # Create an empty array to store the reshaped movie
    new_shape = (num_frames - n_frames, n_frames, y_length, x_length)
    new_dataset: _types.float_like = np.empty(new_shape)

    # Iterate over each frame and create the reshaped movie
    for i in range(num_frames - n_frames):
        new_dataset[i] = dataset[i:i + n_frames]
    return new_dataset