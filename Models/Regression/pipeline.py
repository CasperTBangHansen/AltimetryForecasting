from .. import _types, MainDataClass
from .model import Regression, MetaRegression
from pathlib import Path
from typing import Dict, Any, Tuple

def parse_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool, str]:
    """ Parses kwargs"""
    kwargs_copy = kwargs.copy()
    load: bool = kwargs_copy.pop('load', False)
    save: bool = kwargs_copy.pop('save', True)
    model_name: str = kwargs_copy.pop('name', True)
    return kwargs_copy, load, save, model_name
    
def pipeline_area(
        data: MainDataClass,
        path: Path,
        kwargs: Dict[str, Any]
    ) -> _types.float_like:
    function_kwargs, load, save, model_name = parse_kwargs(kwargs)
    
    if load:
        # Load model
        with open(path / f"{model_name}.pkl", 'rb') as file:
            regressor = MetaRegression.load(file)
    
    else:
        # Create and fit model
        regressor = MetaRegression(Regression, function_kwargs, 0)
        regressor.fit(data.time, data.sla.reshape(data.sla.shape[0], -1))
        if save:
            with open(path / f"{model_name}.pkl", 'wb') as file:
                regressor.save(file)

    # Predict
    return regressor.predict(data.time)