import re
from typing import Mapping
from sklearn import preprocessing
from .. import Protocols, Pipeline

def is_bool(string: str) -> bool:
    """ Checks if a string is a boolean. Ignores upper and lower case"""
    if string.lower() in ('true', 'false'):
        return True
    return False

def to_bool(string: str) -> bool:
    """Convert a string representation of truth to true or false"""
    lower = string.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    raise ValueError(f"Invalid truthy value: {string}")

def is_int(string: str) -> bool:
    """ Checks if the string is an int"""
    if string[0] in ('-', '+'):
        return string[1:].isdigit()
    return string.isdigit()

def is_float(string: str) -> bool:
    """ Checks if a string can be converted to a float"""
    if re.match(r'^-?\d+(?:\.\d+)$', string) is not None:
        return True
    if string.isnumeric():
        return True
    if "/" in string:
        frac = string.split('/')
        return len(frac) == 2 and is_float(frac[0]) and is_float(frac[1])
    return False

def to_float(string: str) -> float:
    """ Converts string to float. Also passes fractions"""
    if "/" in string:
        frac = string.split('/')
        if len(frac) == 2 and frac[0].isnumeric() and frac[1].isnumeric():
            return float(frac[0]) / float(frac[1])
    if is_float(string) or string.isnumeric():
        return float(string)
    raise ValueError("Invalid format of float")

def is_transformer(string: str) -> bool:
    """Checks if the string is a transformer"""
    lower = string.lower()
    if lower == 'standardscaler':
        return True
    return False

def to_transformer(string: str) -> Protocols.BaseTransformer:
    """ Converts string to transformer"""
    lower = string.lower()
    if lower == 'standardscaler':
        return Protocols.SklearnTransformer(preprocessing.StandardScaler())
    raise ValueError("Invalid transformer")

def to_function(string: str, functions: Mapping[str, Pipeline]) -> Pipeline:
    """Converts a string to a function"""
    lower = string.lower()
    if (val := functions.get(lower)) is not None:
        return val
    raise ValueError("Invalid input string")
    