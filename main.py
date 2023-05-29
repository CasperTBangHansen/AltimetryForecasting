from pipeline import pipeline
from Models.AreaClassification import pipeline_area_classification
from Models.Regression import pipeline_area_regression
from Models import Config
from pathlib import Path

def main():
    functions_mapping = {
        'areaclassification': pipeline_area_classification,
        'regression': pipeline_area_regression
    }
    main_config = Config.parse_main_config(Path('main_config.yaml'))
    config = Config.parse_config(Path('pipeline_config.yaml'), functions_mapping)
    pipeline(main_config, config)

if __name__ == '__main__':
    main()