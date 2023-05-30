from pathlib import Path
from typing import Dict, Iterable, Literal, List, Callable, Any
import xarray as xr
import numpy as np
from .. import _types

__all__ = ["MainDataClass", "Pipeline"]

valid_features = Literal['sla', 'sst', 'swh', 'wind_speed']

class MainDataClass:
    
    def __init__(
            self,
            data_path: Path
        ) -> None:
        data = xr.open_dataset(data_path, engine='netcdf4')
        #self.features: _types.float_like = features
        self.lat: _types.float_like = data.Latitude.data[:, 0]
        self.lon: _types.float_like = data.Longitude.data[0]
        self.time: _types.time_like = data.time.data
        self._data = data
        self._cache: Dict[str, _types.float_like | _types.time_like] = {}
        self._data_vars: Dict[valid_features, str] = {'sla': 'sla21', 'sst': 'sst', 'swh': 'swh', 'wind_speed': 'wind_speed'}
    
    @property
    def sla(self) -> _types.float_like:
        """Sea level anomaly in meters"""
        return self._data['sla21'].data * 100
    
    @property
    def sst(self) -> _types.float_like:
        """Sea surface temperature in celsius"""
        return self._data['sst'].data

    @property
    def swh(self) -> _types.float_like:
        """Significant wave height in meters"""
        return self._data['swh'].data
    
    @property
    def wind_speed(self) -> _types.float_like:
        """Wind speed in meters per second"""
        return self._data['wind_speed'].data
    
    def get_features(self, features: Iterable[valid_features]) -> _types.float_like:
        new_features: List[str] = []
        mults: List[float] = []
        for feature in features:
            new_features.append(self._data_vars[feature])
            if new_features[-1] == 'sla':
                mults.append(100)
            else:
                mults.append(1)
        return np.stack([self._data[f].data * mult for f, mult in zip(new_features, mults)])

# Pipeline
Pipeline = Callable[[MainDataClass, Path, Dict[str, Any]], Any]
    
    