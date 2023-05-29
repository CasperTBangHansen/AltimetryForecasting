from .. import Protocols, _types, MainDataClass
from .model import AreaClassification
from sklearn import preprocessing
from pathlib import Path
from typing import Dict, Any, Tuple

def parse_kwargs(kwargs: Dict[str, Any]) -> Tuple[int, int, int, bool, bool, Protocols.BaseTransformer]:
    """ Parses kwargs"""
    som_m: int = kwargs.get('som_m', 2)
    som_n: int = kwargs.get('som_n', 2)
    epochs: int = kwargs.get('epochs', 10)
    load: bool = kwargs.get('load', False)
    save: bool = kwargs.get('save', True)
    transformer: Protocols.BaseTransformer = kwargs.get('transformer', Protocols.SklearnTransformer(preprocessing.StandardScaler()))
    return som_m, som_n, epochs, load, save, transformer
    
def pipeline(
        data: MainDataClass,
        path: Path,
        kwargs: Dict[str, Any]
    ) -> _types.int_like:
    som_m, som_n, epochs, load, save, transformer = parse_kwargs(kwargs)
    if load:
        # Model model
        with open(path, 'rb') as file:
            classifier = AreaClassification.load(file)
    else:
        # Create and fit model
        classifier = AreaClassification(transformer, som_m, som_n, epochs)
        classifier.fit(data.sla)
        if save:
            with open(path, 'wb') as file:
                classifier.save(file)

    # Predict
    return classifier.predict(data.sla)