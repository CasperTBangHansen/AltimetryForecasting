from .. import _types, MainDataClass
from .model import Regression, MetaRegression
from pathlib import Path
from typing import Dict, Any, Tuple

def parse_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
    """ Parses kwargs"""
    kwargs_copy = kwargs.copy()
    load: bool = kwargs_copy.pop('load', False)
    save: bool = kwargs_copy.pop('save', True)
    return kwargs_copy, load, save
    
def pipeline_area(
        data: MainDataClass,
        path: Path,
        kwargs: Dict[str, Any]
    ) -> _types.float_like:
    kwargs, load, save = parse_kwargs(kwargs)
    
    if load:
        # Load model
        with open(path, 'rb') as file:
            regressor = MetaRegression.load(file)
    
    else:
        # Create and fit model
        regressor = MetaRegression(Regression, kwargs, 1)
        regressor.fit(data.time, data.sla.reshape(data.sla.shape[0], -1))
        if save:
            with open(path, 'wb') as file:
                regressor.save(file)

    # Predict
    return regressor.predict(data.time)