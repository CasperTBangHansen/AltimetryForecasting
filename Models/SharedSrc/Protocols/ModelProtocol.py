from typing import Protocol, Self, BinaryIO
from .. import _types

__all__ = ["FitModelRegressor", "FitModelClassifier", "SaveModel", "Statistics", "BaseRegressor", "BaseClassifier", "NotFittedError"]


class FitModelRegressor(Protocol):
    """Protocal for fitting and prediciting using the model"""

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        ...

    def fit(self, x: _types.float_like, y: _types.float_like) -> None:
        """Fits the model x ~ y"""
        ...
    
    def predict(self, x: _types.float_like) -> _types.float_like:
        """ Makes a prediction using x"""
        ...

class FitModelClassifier(Protocol):
    """Protocal for fitting and prediciting using the model"""

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        ...

    def fit(self, x: _types.float_like) -> None:
        """Fits the model"""
        ...
    
    def predict(self, x: _types.float_like) -> _types.int_like:
        """ Makes a prediction on X"""
        ...

class Statistics(Protocol):
    """Protocol for computing the statistics"""

    def summary(self) -> str:
        """ Creates a summary of the model"""
        ...


class SaveModel(Protocol):
    """ Load and save model"""

    def save(self, file: BinaryIO) -> None:
        """Saves the model to a file"""
        ...
    
    @classmethod
    def load(cls, file: BinaryIO) -> Self:
        """Loads the model from a file"""
        ...

class BaseRegressor(FitModelRegressor, Statistics, SaveModel, Protocol):
    """ Base regressor implementation"""
    ...

class BaseClassifier(FitModelClassifier, Statistics, SaveModel, Protocol):
    """ Base classifier implementation"""
    ...

class NotFittedError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)