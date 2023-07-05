from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

__all__ = ["DatasetParameters", "Loss"]

@dataclass
class DatasetParameters:
    """ Main hyperparameters to construct the dataset"""
    batch_size: int
    sequence_length: int
    sequence_steps: int
    prediction_steps: int
    fill_nan: float
    train_end: date
    validation_end: date
    n_predictions: int
    train_start: Optional[date] = None

@dataclass
class Loss:
    """Maps epoch id to training and validation loss"""
    epoch: int
    training_loss: float
    validation_loss: float
    time: datetime
    learning_rate: float