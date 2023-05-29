from Models import Config, MainDataClass
from typing import List

def pipeline(main_config: Config.MainConfig, stages: List[Config.Stage]):
    
    # Load data
    data = MainDataClass(main_config.paths.grid_path)

    for stage in stages:
        if stage.name != 'fit':
            continue
        print(f"Stage: {stage.name}")
        for task in stage.tasks:
            print(f"\tTask: {task.name}")
            for process in task.processes:
                process.result = process.function(data, main_config.paths.saved_models, process.kwargs)
            print("")
        print("")