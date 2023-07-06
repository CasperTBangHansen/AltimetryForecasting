import TrainingLoopEncoderDecoder
from torch.utils.data import DataLoader
from .Shared import DatasetParameters, SLADataset
import torch
import numpy as np
from typing import Tuple

try:
    from ..Models import _types
    from ..Models.Regression.model import MetaRegression, Regression
except:
    from Models import _types
    from Models.Regression.model import MetaRegression, Regression

def loader_prediction(model, loader: DataLoader[SLADataset], device: torch.device, n_predictions: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Make predictions
    model = model.to(device)
    features, *_ = next(iter(loader))
    outputs, targets, target_times = TrainingLoopEncoderDecoder.predict_seq(
        model,
        loader,
        tuple(features.shape[-2:]),
        device,
        n_predictions
    )
    target_times = (np.array([0], dtype="datetime64[ns]") + target_times.astype("timedelta64[D]")).astype("datetime64[D]")
    return outputs, targets, target_times

def rmse_pixelwise(targets: _types.float_like, outputs: _types.float_like):
    diffs: _types.float_like = targets - outputs
    rmses = np.full(diffs.shape[1:], np.nan)
    for lat_idx in range(targets.shape[2]):
        for lon_idx in range(targets.shape[3]):
            pixel_diffs = diffs[:, :, lat_idx, lon_idx]
            for i, pixel in enumerate(pixel_diffs.T):
                pixel_diffs_nonan = pixel[~np.isnan(pixel)]
                if len(pixel_diffs_nonan) == 0:
                    continue
                rmses[i, lat_idx, lon_idx] = np.sqrt(pixel_diffs_nonan @ pixel_diffs_nonan/len(pixel_diffs_nonan))
    return diffs, rmses

def get_pred_title_multi(dataset_parameters: DatasetParameters, n_predictions: int):
    return f"{n_predictions * dataset_parameters.prediction_steps} days predictions using {dataset_parameters.sequence_length} day(s) of data\nwith {dataset_parameters.sequence_steps - 1} day(s) inbetween"

def get_pred_title(dataset_parameters: DatasetParameters, n_predictions: int):
    return get_pred_title_multi(dataset_parameters, n_predictions).replace('\n', ' ')