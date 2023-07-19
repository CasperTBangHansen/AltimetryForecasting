import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset
from . import DatasetParameters
from AltimeterAutoencoder.src import _types

class SLADataset(Dataset):
    """Constructs the SLA dataset"""
    def __init__(
        self,
        x: _types.float_like,
        t: _types.int_like,
        datasetparameters: DatasetParameters,
        min_value = None,
        difference = None
    ):
        # Convert to float32 (float64 is overkill)
        x_32 = x.astype(np.float32)
        self.min_value = min_value
        self.difference = difference
        if min_value is not None and difference is not None:
            # Feature scale between -1 and 1
            min_feature_scale = -1
            max_feature_scale = 1
            difference_feature_scale = (max_feature_scale - min_feature_scale)
            x_32 = min_feature_scale + difference_feature_scale*(x_32 - self.min_value) / self.difference
        
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