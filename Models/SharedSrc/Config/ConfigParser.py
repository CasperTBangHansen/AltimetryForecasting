from typing import Dict, Any, List, Mapping
from pathlib import Path
from datetime import date
from yaml import safe_load
from .. import Pipeline
from .ConfigTypes import MainConfig, Versions, Paths, _Function, _Macro, Task, Process, Stage
from .Converter import to_function

__all__ = ["parse_config", "parse_main_config"]

def parse_process(process: Dict[str, Any], functions: Mapping[str, Pipeline]) -> Process:
    """ Converts dict to process"""
    process_name: str = process.pop('Type')
    function = to_function(process_name, functions)
    return Process(process = process_name, kwargs = process, function=function)

def parse_task(name: str, task: List[Dict[str, Any]], functions: Mapping[str, Pipeline]) -> Task:
    """ Converts dict to task"""
    processes = [parse_process(process, functions) for process in task]
    return  Task(name = name, processes=processes)

def parse_stage(name: str, stage: Dict[str, bool | List[Dict[str, Any]]], functions: Mapping[str, Pipeline]) -> Stage:
    """ Converts dict to stage"""
    # Check if the stage can be performed in parallel
    if 'parallel' in stage:
        parallel = stage.pop('parallel')
        if not isinstance(parallel, bool):
            parallel = False
    else:
        parallel = False

    # Get tasks
    tasks = [parse_task(name, task, functions) for name, task in stage.items() if isinstance(task, list)]
    
    # Construct stage
    return Stage(name = name, parallel = parallel, tasks = tasks)

def parse_pipeline(initial_pipeline: Dict[str, Any], functions: List[_Function], macros: List[_Macro]):
    macro_names = [m.name for m in macros]

    def traverse_pipeline(pipeline: Dict[str, Any] | Any | List[Any]):
        if isinstance(pipeline, list):
            pipeline_list = []
            for p in pipeline:
                traversed_pipeline = traverse_pipeline(p)
                if isinstance(traversed_pipeline, list):
                    pipeline_list.extend(traversed_pipeline)
                else:
                    pipeline_list.append(traversed_pipeline)
            return pipeline_list

        if isinstance(pipeline, dict):
            # Check for macro
            if 'macro' in pipeline:
                # Apply macro
                macro_name = pipeline['macro'].split('(')[0]
                if macro_name in macro_names:
                    macro_index = macro_names.index(macro_name)
                    return traverse_pipeline(macros[macro_index].apply_macro(pipeline['macro']))
            
            # Run through the rest of the pipeline nodes
            current_pipeline = pipeline
            for key, value in pipeline.items():
                if key == 'Type':
                    continue
                current_pipeline[key] = traverse_pipeline(value)
            return current_pipeline

        # Parse function
        elif isinstance(pipeline, str):
            for function in functions:
                if function.name in pipeline:
                    return traverse_pipeline(function.apply_function(pipeline))
        
        # Return value if it is not a list or dict
        return pipeline
    new_pipeline = traverse_pipeline(initial_pipeline)
    return new_pipeline

def parse_config(path: Path, functions_mapping: Mapping[str, Pipeline]) -> List[Stage]:
    # Load config
    with open(path, 'rb') as file:
        config = safe_load(file)
    
    # Extract functions and macros
    functions = [_Function(function) for function in config['functions']]
    macros = [_Macro(header, body) for macro in config['macros'] for header, body in macro.items()]

    parse_pipeline(config['pipeline'], functions, macros)
    # Process stages
    stages: List[Stage] = []
    for name, stage in config['pipeline'].items():
        stages.append(parse_stage(name, stage, functions_mapping))
    return stages

def parse_main_config(path: Path) -> MainConfig:
    # Load config
    with open(path, 'rb') as file:
        config = safe_load(file)
    
    # Convert string to path
    for key, value in config['paths'].items():
        config['paths'][key] = Path(value)

    # Convert YYYY-MM-DD to date
    train_test_split = date(*list(map(int, config['train_test_split'].split('-'))))
    
    # Construct main config object
    main_config =  MainConfig(
        Versions(**config['versions']),
        Paths(**config['paths']),
        train_test_split
    )
    return main_config