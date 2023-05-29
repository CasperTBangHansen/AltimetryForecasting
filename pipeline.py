from Models import _types, Config, MainDataClass
from typing import List, Tuple
import xarray as xr
from pathlib import Path

def pipeline(data_path: Path, stages: List[Config.Stage]):
    
    # Load data
    data = MainDataClass(data_path)

    for stage in stages:
        print(f"Stage: {stage.name}")
        for task in stage.tasks:
            print(f"\tTask: {task.name}")
            for process in task.processes:
                kwargs_str = ", ".join([str(v) for v in process.kwargs.values()])
                print(f"\t\tProcess: {process.process}({kwargs_str})")
            print("")
        print("")