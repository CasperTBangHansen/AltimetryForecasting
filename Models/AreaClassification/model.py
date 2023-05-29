import numpy as np
from sklearn_som.som import SOM
import pickle
from typing import Self, BinaryIO
from .. import Protocols, _types


class AreaClassification:
    """ Classifices each feature to a class set of classes n x m"""

    def __init__(self, transformer: Protocols.BaseTransformer, som_m: int = 2, som_n: int = 2, epochs: int = 10) -> None:
        self._transformer = transformer
        self.som_m = som_m
        self.som_n = som_n
        self.epochs = epochs
        self._som: SOM | None = None
        self._trained = False

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        return self._trained

    @staticmethod
    def _construct_features(features: _types.float_like) -> _types.float_like:
        """ Construct mean/std features"""
        _mean: _types.float_like = np.nanmean(features, axis=0)
        _std: _types.float_like = np.nanstd(features, axis=0)

        nans: _types.bool_like = np.isnan(_mean)
        _mean = _mean[~nans]
        _std = _std[~nans]
        return np.vstack([_std, _mean]).T # type: ignore

    def fit(self, features: _types.float_like) -> None:
        """ Fits the model using the input features"""
        # Feature scale features
        _features = self._construct_features(features)
        self._transformer.fit(_features)
        _features_scaled = self._transformer.transform(_features)

        # Construct SOM
        self._som = SOM(
            m=self.som_m,
            n=self.som_n,
            dim=_features_scaled.shape[1],
            max_iter=_features_scaled.shape[0] * (1 + self.epochs)
        )

        # Fit som
        self._som.fit(_features_scaled, epochs = self.epochs)
        self._trained = True

    def predict(self, features: _types.float_like) -> _types.int_like:
        """ Predicts using the fitted model"""
        if not self._trained or self._som is None or self._transformer is None:
            raise Protocols.NotFittedError("Fit must be called before predict")
        _features_scaled = self._transformer.transform(features)
        return self._som.predict(_features_scaled)
    
    def save(self, file: BinaryIO) -> None:
        """ Saves model to a file"""
        pickle.dump(self.__dict__, file)

    @classmethod
    def load(cls, file: BinaryIO) -> Self:
        """ Loads the model from a file"""
        return pickle.load(file)