import numpy as np
import torch
from typing import Tuple
try:
    from ..Models import _types
except:
    from Models import _types
from torch.utils.data import Dataset

class SLADataset(Dataset):
    def __init__(self, x: _types.float_like, sequence_length: int, prediction_length: int, fill_nan: float):
        # Convert to float32 (float64 is overkill)
        x_32 = x.astype(np.float32)
        
        self.mask = torch.tensor(np.isnan(x_32), dtype=torch.bool)
        self.x = torch.nan_to_num(torch.tensor(x_32, dtype=torch.float32), nan = fill_nan)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.fill_nan = fill_nan

    def __len__(self) -> int:
        return len(self.x) - self.sequence_length - self.prediction_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.x[idx:idx + self.sequence_length]
        target = self.x[idx + self.sequence_length + self.prediction_length - 1]
        mask = self.mask[idx + self.sequence_length + self.prediction_length - 1]
        return features.unsqueeze(0), target, mask
    
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