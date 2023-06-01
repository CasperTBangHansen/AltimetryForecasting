import numpy as np
from ..Models import _types

def stack_frames(dataset: _types.float_like, n_frames: int) -> _types.float_like:
    """
    Stacks each frame in the dataset into n_frames.
    Converts dataset of shape
        (num_frames, y_length, x_length)
    to
        (num_frames - n_frames, n_frames, y_length, x_length)
    """
    num_frames, y_length, x_length = dataset.shape

    # Create an empty array to store the reshaped movie
    new_shape = (num_frames - n_frames, n_frames, y_length, x_length)
    new_dataset: _types.float_like = np.empty(new_shape)

    # Iterate over each frame and create the reshaped movie
    for i in range(num_frames - n_frames):
        new_dataset[i] = dataset[i:i + n_frames]
    return new_dataset