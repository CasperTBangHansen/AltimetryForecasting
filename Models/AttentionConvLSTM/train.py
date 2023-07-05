import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import sys
from datetime import date
from typing import List
import xarray as xr
import numpy as np
from . import Seq2SeqAttention, SaveLoadModels, InputModel, OutputModel, Attention, Encoder, Decoder
from .Loss import create_masked_loss_function_diff
from .training_loop import train_validation_loop
from ..Shared import DatasetParameters, Loss
from ..Shared.Dataloader import SLADataset
from AltimeterAutoencoder.src.regressor import MetaRegression, fit_regressor
from AltimeterAutoencoder.src import _types

def print_to_file_on_epoch(loss: Loss, _: _types.float_like, __: _types.float_like, n_epochs: int):
    logger = logging.getLogger()
    logger.info(f"Epoch: {n_epochs}/{loss.epoch} Training Loss: {loss.training_loss:.5f} Validation Loss: {loss.validation_loss:.5f}")

def train_model():
    BASEPATH = Path(r"Data/Grids/")
    SAVEFOLDER = Path("SavedModels", "ConvLSTMAttention_LastDayInput_32Channels_grad_loss")
    MODEL_NAME = "checkpoint_20.pkl"
    SAVEFOLDER.mkdir(parents=True, exist_ok=True)
    LOAD_MODEL = True

    logging.basicConfig(
        filename = SAVEFOLDER / 'log.log',
        level = logging.DEBUG,
        format = '%(asctime)s:%(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Model parameters
    kernel_size = (3, 3)
    padding = ((kernel_size[0] - 1)//2, (kernel_size[1] - 1)//2)
    frame_size = (128, 360)
    in_channels = 1
    encoder_in_channels = 16
    encoder_out_channels = 32

    # Training parameters
    learning_rate = 1e-6 #1e-5
    weight_decay = 0.0001
    n_sequences = 30
    scheduler = None
    epochs = 500
    save_every = 10
    teacher_forcing_len = 10
    teacher_forcing_ratio = 0.5
    teacher_forcing_ratio_list: List[float] = [0]*epochs

    # Loss
    criterion = create_masked_loss_function_diff(nn.MSELoss)


    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    regressor = None
    if LOAD_MODEL:
        model, optimizer, last_loss, dataset_parameters = SaveLoadModels.load_checkpoint(
            SAVEFOLDER / MODEL_NAME,
            Seq2SeqAttention,
            DEVICE
        )
        with open(SAVEFOLDER / "Regression.pkl", 'rb') as file:
            regressor = MetaRegression.load(file)
        start_epoch = last_loss.epoch
    else:
        dataset_parameters = DatasetParameters(
            batch_size = 1,
            sequence_length = 30,
            sequence_steps = 1,
            prediction_steps = 1,
            fill_nan = 0,
            train_end = date(2014, 1, 1),
            validation_end = date(2019, 1, 1),
            n_predictions = 30,
        )
        start_epoch = 0

        # Construct models
        input_model = InputModel(in_channels, encoder_in_channels)

        encoder = Encoder(
            in_channels = encoder_in_channels,
            out_channels = encoder_out_channels,
            kernel_size = kernel_size,
            padding = padding,
            activation = nn.Tanh(),
            frame_size = frame_size,
        )

        attention = Attention(encoder_out_channels, encoder_out_channels)

        decoder = Decoder(
            in_channels = encoder_in_channels,
            hidden_channels = encoder_out_channels,
            out_channels = encoder_out_channels,
            kernel_size = kernel_size,
            padding = padding,
            activation = nn.Tanh(),
            frame_size = frame_size,
            attention = attention
        )

        output_network = OutputModel(
            in_channels = decoder.resulting_channels,
            hidden_channels = [32, 16, 8],
            output_channels = in_channels
        )
        model = Seq2SeqAttention(input_model, encoder, decoder, output_network)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    with xr.open_dataset(BASEPATH / "without_polar_v5_mss21.nc", engine="netcdf4") as file:
        file = file.sortby('time')
        sla = file['sla21'].data[:, :-1]
        times: _types.time_like = file['time'].data
    train_start = date(2006, 1, 1)

    # Set train, validation and test intervals
    train_start_np = np.array(train_start).astype("datetime64[ns]")
    train_end = np.array(dataset_parameters.train_end).astype("datetime64[ns]")
    validation_end = np.array(dataset_parameters.validation_end).astype("datetime64[ns]")

    # Save times
    bool_train = (times > train_start_np) & (times <= train_end)
    bool_validation = (times > train_end) & (times <= validation_end)
    
    if not LOAD_MODEL:
        regressor = fit_regressor(times[bool_train], sla[bool_train], SAVEFOLDER / "Regression.pkl")

    if regressor is not None:
        sla -= regressor.predict(times).reshape(*sla.shape)
    train_time: _types.int_like = times[bool_train].astype("datetime64[D]").astype(int)
    validation_time: _types.int_like = times[bool_validation].astype("datetime64[D]").astype(int)

    # Save sla features
    train_features = sla[bool_train]
    validation_features = sla[bool_validation]

    # Kwargs to dataloaders
    kwargs_dataloader = {
        'shuffle': False,
        'batch_size': dataset_parameters.batch_size
    }

    # Dataloders
    train_loader = DataLoader(SLADataset(train_features, train_time, dataset_parameters), **kwargs_dataloader)
    validation_loader = DataLoader(SLADataset(validation_features, validation_time, dataset_parameters), **kwargs_dataloader)

    # Train
    scheduler = None
    losses = train_validation_loop(
        model = model,
        train_loader = train_loader,
        val_loader = validation_loader,
        criterion = criterion,
        optimizer = optimizer,
        num_epochs = epochs,
        start_epoch = start_epoch,
        device = DEVICE,
        update_function = print_to_file_on_epoch,
        path = SAVEFOLDER,
        save_n_epochs = save_every,
        dataset_parameters = dataset_parameters,
        teacher_forcing_ratios = teacher_forcing_ratio_list,
        losses = None,
        scheduler = scheduler,
        n_sequences = n_sequences,
    )


if __name__ == '__main__':
    train_model()