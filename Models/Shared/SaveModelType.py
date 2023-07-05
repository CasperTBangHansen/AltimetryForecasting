from typing import Protocol, Dict, Any, Mapping, Callable
import sys
if sys.version_info >= (3,11):
    from typing import Self
else:
    Self = None

class SaveModel(Protocol):
    """Protocol for a model to be saved"""

    def get_kwargs(self) -> Dict[str, Any]:
        """Saves the input arguments of the class"""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dict of the model"""
        ...
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any:
        """Sets the statedict of the model"""
        ...
    
    forward: Callable[..., Any]
    
    def __call__(self, *args: Any) -> Any:
        """Forward function for pytorch nn.Module"""
        ...
    
    def train(self, mode: bool = True) -> Self:
        """ Convert model to training mode"""
        ...

    def eval(self) -> Self:
        """ Convert model to evalution mode"""
        ...