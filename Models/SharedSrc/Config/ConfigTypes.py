from pydantic.dataclasses import dataclass
from pydantic import Field
from pathlib import Path
from datetime import date
import ast
from typing import List, Tuple, Dict, Any
from . import Converter
from .. import Pipeline


def parse_kwargs(kwargs: Dict[str, str]) -> Dict[str, Any]:
    """ Converts string to the correct type"""
    kwargs_copy: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if not isinstance(value, str):
            kwargs_copy[key] = value
        elif Converter.is_int(value):
            kwargs_copy[key] = int(value)
        elif Converter.is_float(value):
            kwargs_copy[key] = Converter.to_float(value)
        elif Converter.is_bool(value):
            kwargs_copy[key] = Converter.to_bool(value)
        elif Converter.is_transformer(value):
            kwargs_copy[key] = Converter.to_transformer(value)
        else:
            try:
                kwargs_copy[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                kwargs_copy[key] = value
    return kwargs_copy

def custom_split(string: str, left: str, right: str, split_on: str) -> Tuple[str,...]:
    """
    Splits a string based on the split_on character but only when
    there has been an equal amount of left characters and right characters
    """
    can_split = 0
    split_str = []
    last_split = 0
    for current_pos, chr in enumerate(string):
        if chr == left:
            can_split += 1
        elif chr == right:
            can_split -= 1
        elif chr == split_on and can_split == 0:
            split_str.append(string[last_split:current_pos])
            last_split = current_pos
            if current_pos != 0:
                last_split += 1
    return tuple(split_str)

def parse_argument(template: str) -> Tuple[str, ...]:
    """Parses arguments from config file"""
    # Get start and end indexes
    param_start = template.find('(')
    param_end = template.rfind(')')
    arguments = template[param_start + 1:param_end].replace(' ', '')

    # Check for tuple
    if "(" in arguments and ")" in arguments:
        return custom_split(arguments, "(", ")", ",")
    # Check for list
    if "[" in arguments and "]" in arguments:
        return custom_split(arguments, "[", "]", ",")
    return tuple(arguments.split(','))

@dataclass
class Versions:
    forecasting: int

@dataclass
class Paths:
    grid_path: Path
    saved_models: Path

@dataclass
class MainConfig:
    versions: Versions
    paths: Paths
    train_test_split: date

@dataclass
class _Function:
    function: str
    name: str = Field(default_factory=lambda: "")
    arguments: Tuple[str, ...] = Field(default_factory=lambda: ("",))

    def __post_init__(self):
        self.name = self.function.split('(')[0]
        self.arguments = parse_argument(self.function)

    def apply_function(self, match: str) -> Dict[str, Any]:
        arguments = parse_argument(match)
        out = {'Type': self.name}
        for name, argument in zip(self.arguments, arguments):
            out[name] = argument
        return out

@dataclass
class _Macro:
    template: str
    functions: List[str]
    name: str = Field(default_factory=lambda: "")
    arguments: Tuple[str, ...] = Field(default_factory=lambda: ("",))
    parsed_functions: List[str] = Field(default_factory=lambda: [])

    def __post_init__(self):
        self.name = self.template.split('(')[0]
        self.arguments = parse_argument(self.template)
        self.parsed_functions = self._parsed_functions()

    def apply_macro(self, match: str) -> List[str]:
        arguments = parse_argument(match)
        macros: List[str] = []
        for function in self.parsed_functions:
            mapping = dict(zip(self.arguments, arguments))
            macros.append(function.format(**mapping))
        return macros

    def _parsed_functions(self) -> List[str]:
        parsed_functions: List[str] = []
        for function in self.functions:
            parsed_function = function
            for argument in self.arguments:
                parsed_function = parsed_function.replace(argument, f'{{{argument}}}')
            parsed_functions.append(parsed_function)
        return parsed_functions

@dataclass
class Process:
    process: str
    kwargs: Dict[str, Any]
    function: Pipeline
    result: Any = Field(default_factory=lambda: None)

    def __post_init__(self,):
        self.kwargs = parse_kwargs(self.kwargs)

@dataclass
class Task:
    name: str
    processes: List[Process]

@dataclass
class Stage:
    name: str
    parallel: bool
    tasks: List[Task]
