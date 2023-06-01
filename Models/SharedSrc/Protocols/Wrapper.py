from . import BaseTransformer, NotFittedError
from .. import _types
import pickle
from typing import BinaryIO
import sys
if sys.version_info >= (3,11):
    from typing import Self
else:
    Self = None
from sklearn.base import TransformerMixin

__all__ = ["SklearnTransformer"]

class SklearnTransformer(BaseTransformer):
    def __init__(self, transformer: TransformerMixin) -> None:
        self._transformer: TransformerMixin = transformer
        self._trained: bool = False

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        return self._trained

    def fit(self, features: _types.float_like) -> None:
        """Fits the model"""
        self._transformer.fit(features) # type: ignore
        self._trained = True
    
    def transform(self, features: _types.float_like) -> _types.float_like:
        """ Makes a prediction on X"""
        if not self._trained:
            raise NotFittedError("Fit must be called before predict")
        return self._transformer.transform(features) # type: ignore

    def save(self, file: BinaryIO) -> None:
        """ Saves model to a file"""
        pickle.dump(self.__dict__, file)

    @classmethod
    def load(cls, file: BinaryIO) -> Self:
        """ Loads the model from a file"""
        return pickle.load(file)
    
    
