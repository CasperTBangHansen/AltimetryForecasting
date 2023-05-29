from Models import _types, Config
from typing import List, Tuple
import xarray as xr
from numpy.typing import NDArray
import numpy as np
from pathlib import Path

def load_data(data_path: Path) -> Tuple[_types.float_like, _types.float_like, NDArray[np.datetime64], _types.float_like]:
    """ Load netcdf file and gets the attributes"""
    data = xr.open_dataset(data_path, engine='netcdf')
    lats = data.Latitude
    lons = data.Longitude
    return 0 # type: ignore

def pipeline(data_path: Path, stages: List[Config.Stage]):
    
    # Load data
    #lat, lon, time, features = load_data(data_path)

    for stage in stages:
        print(f"Stage: {stage.name}")
        for task in stage.tasks:
            print(f"\tTask: {task.name}")
            for process in task.processes:
                kwargs_str = ", ".join([str(v) for v in process.kwargs.values()])
                print(f"\t\tProcess: {process.process}({kwargs_str})")
            print("")
        print("")