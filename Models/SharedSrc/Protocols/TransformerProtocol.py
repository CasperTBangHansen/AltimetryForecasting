from typing import Protocol
from . import ModelProtocol
from .. import _types

class Transformer(Protocol):
    """Protocal for fitting and prediciting using the model"""

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        ...

    def fit(self, features: _types.float_like) -> None:
        """Fits the model"""
        ...
    
    def transform(self, features: _types.float_like) -> _types.float_like:
        """ Makes a prediction on X"""
        ...

class BaseTransformer(Transformer, ModelProtocol.SaveModel, Protocol):
    """ Base transformer implementation"""
    ...
