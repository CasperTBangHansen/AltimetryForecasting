import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset
from ..Shared import DatasetParameters, SLADataset
from AltimeterAutoencoder.src import _types

class SLADatasetEncoded(Dataset):
    """Constructs the SLA dataset"""
    def __init__(
        self,
        feature_encodings,
        target_encodings,
        dataset,
        index_mapping,
        mins = None,
        differences = None,
        min_feature_space = -1,
        max_feature_space = 1
    ):
        if mins is None or differences is None:
            means = torch.mean(feature_encodings, dim=(0,2))
            stds = torch.std(feature_encodings, dim=(0,2))
            upper = means + 2 * stds
            lower = means - 2 * stds
            valid_feature_encodings = feature_encodings * (
                (feature_encodings < upper.unsqueeze(0).unsqueeze(2)) &
                (feature_encodings > lower.unsqueeze(0).unsqueeze(2))
            )
            maxs = torch.amax(valid_feature_encodings, dim=(0,2)).unsqueeze(0).unsqueeze(2)
            self.mins = torch.amin(valid_feature_encodings, dim=(0,2)).unsqueeze(0).unsqueeze(2)
            self.differences = (maxs - self.mins).clip(1e-9)
        else:
            self.mins = mins
            self.differences = differences
        difference_feature_scale = (max_feature_space - min_feature_space)
        self.feature_encodings = min_feature_space + difference_feature_scale*(feature_encodings - self.mins) / self.differences
        self.target_encodings = min_feature_space + difference_feature_scale*(target_encodings - self.mins) / self.differences
        
        self.dataset = dataset
        self.index_mapping = index_mapping
        self._len = len(self.feature_encodings)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Features
        features = self.feature_encodings[idx]
        target = self.target_encodings[idx]
        return features, target

def outlier_detect(
    encoder,
    decoder,
    loader,
    device,
    threshold_cm = 0.6,
    mins = None,
    differences = None
) -> SLADatasetEncoded:
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        encodings = []
        encoding_targets = []
        valid_indexes = []
        indexes = torch.arange(len(loader.dataset))
        current_idx = 0
        for i, (features, target, mask, features_time, pred_time) in enumerate(loader):

            # Features
            in_features = features.to(device)
            encoding = encoder(in_features.transpose(1,2))
            result = decoder(encoding).transpose(1, 2)
            rmse_features = (((in_features - result) * 100)**2).mean(dim=(1,2,3,4)).sqrt().cpu()
            
            # Features
            in_features = target.unsqueeze(1).to(device)
            encoding_target = encoder(in_features.transpose(1,2))
            result = decoder(encoding_target).transpose(1, 2)
            rmse_targets = (((in_features - result) * 100)**2).mean(dim=(1,2,3,4)).sqrt().cpu()
            bool_arr = (rmse_features <= threshold_cm) & (rmse_targets <= threshold_cm)
            
            next_idx = current_idx + len(features)
            valid_indexes.append(indexes[current_idx:next_idx][bool_arr])
            current_idx = next_idx
            
            encodings.append(encoding.transpose(1, 2).cpu()[bool_arr])
            encoding_targets.append(encoding_target.transpose(1, 2).cpu()[bool_arr])

        encodings = torch.vstack(encodings)
        encoding_targets = torch.vstack(encoding_targets)
        valid_indexes = torch.hstack(valid_indexes)
        
        return SLADatasetEncoded(
            encodings,
            encoding_targets,
            loader.dataset,
            valid_indexes,
            mins = mins,
            differences = differences
        )