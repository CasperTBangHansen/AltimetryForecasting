import logging
import sys
from pathlib import Path
from datetime import date
from typing import Tuple, List
import numpy as np
import xarray as xr
import torch
from torch import nn
from torch.utils.data import DataLoader
from AltimeterAutoencoder.src.save_load_model import load_model as load_autoencoder
from AltimeterAutoencoder.src.regressor import MetaRegression
from AltimeterAutoencoder.src import _types
from AltimeterAutoencoder.src.regressor import fit_regressor, MetaRegression
from . import SaveLoadModels
from .training_loop import train_validation_loop
from ..Shared import Loss, DatasetParameters
from ..Shared.Dataloader import SLADataset
from ..AttentionConvLSTM import Seq2SeqAttention, InputModel, OutputModel, Attention, Encoder, Decoder
from ..AttentionConvLSTM.Loss import create_masked_loss_function_diff

def print_to_file_on_epoch(loss: Loss, _: _types.float_like, __: _types.float_like, n_epochs: int):
    logger = logging.getLogger()
    logger.info(f"Epoch: {n_epochs}/{loss.epoch} Training Loss: {loss.training_loss:.5f} Validation Loss: {loss.validation_loss:.5f}")

def setup_model(
    in_channels: int,
    encoder_in_channels: int,
    encoder_out_channels: int,
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
    frame_size: Tuple[int, int],
    hidden_output_network_channels: List[int]
) -> Seq2SeqAttention:
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
        hidden_channels = hidden_output_network_channels,
        output_channels = in_channels
    )
    return Seq2SeqAttention(input_model, encoder, decoder, output_network)

def load_model(
    model_path: Path,
    device: torch.device,
    regressor_path: Path
) -> Tuple[Seq2SeqAttention, torch.optim.Optimizer, Loss, DatasetParameters, MetaRegression, int, str]:
    model, optimizer, last_loss, dataset_parameters, auto_encoder_name = SaveLoadModels.load_checkpoint(
        model_path,
        Seq2SeqAttention,
        device
    )
    with open(regressor_path, 'rb') as file:
        regressor = MetaRegression.load(file)
    start_epoch = last_loss.epoch
    return model, optimizer, last_loss, dataset_parameters, regressor, start_epoch, auto_encoder_name

def setup_data(
    datapath: Path,
    save_path: Path,
    dataset_parameters: DatasetParameters,
    regressor: MetaRegression | None = None
) -> Tuple[DataLoader[SLADataset], DataLoader[SLADataset]]:
    with xr.open_dataset(datapath, engine="netcdf4") as file:
        file = file.sortby('time')
        sla = file['sla21'].data[:, :-1]
        times: _types.time_like = file['time'].data

    # Set train, validation and test intervals
    train_start_np = np.array(dataset_parameters.train_start).astype("datetime64[ns]")
    train_end = np.array(dataset_parameters.train_end).astype("datetime64[ns]")
    validation_end = np.array(dataset_parameters.validation_end).astype("datetime64[ns]")

    # Save times
    bool_train = (times > train_start_np) & (times <= train_end)
    bool_validation = (times > train_end) & (times <= validation_end)

    if regressor is None:
        regressor = fit_regressor(times[bool_train], sla[bool_train], save_path)

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
    return train_loader, validation_loader

def train_model():
    DATAPATH = Path(r"Data/Grids/")
    SAVEFOLDER = Path("SavedModels", "ConvLSTMAttentionEncDec")
    AUTOENCODERFOLDER = Path("SavedModels", "AutoEncoder")
    auto_encoder_name = "checkpoint_20.pkl"
    MODEL_NAME = "checkpoint_20.pkl"
    SAVEFOLDER.mkdir(parents=True, exist_ok=True)
    LOAD_MODEL = True

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logging.basicConfig(
        filename = SAVEFOLDER / 'log.log',
        level = logging.DEBUG,
        format = '%(asctime)s:%(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Dataset parameters
    default_batch_size = 1
    default_sequence_length = 30
    default_sequence_steps = 1
    default_prediction_steps = 1
    default_n_predictions = 30

    # Model parameters
    kernel_size = (3, 3)
    padding = ((kernel_size[0] - 1)//2, (kernel_size[1] - 1)//2)
    frame_size = (128, 360)
    encoder_in_channels = 16
    encoder_out_channels = 32

    # Training loop
    learning_rate = 1e-6 #1e-5
    weight_decay = 0.0001
    epochs = 100
    save_every = 10
    n_sequences = 30
    scheduler = None
    teacher_forcing_ratio_list: List[float] = [0]*epochs

    # Loss
    criterion = create_masked_loss_function_diff(nn.MSELoss)

    if LOAD_MODEL:
        model, optimizer, last_loss, dataset_parameters, regressor, start_epoch, auto_encoder_name = load_model(
            SAVEFOLDER / MODEL_NAME,
            DEVICE,
            SAVEFOLDER / "Regression.pkl"
        )

    # Load autoencoder
    encoder, decoder, _, _, _, fill_nan, _, train_start, train_end, validation_end = load_autoencoder(
        AUTOENCODERFOLDER / auto_encoder_name,
        DEVICE
    )

    if not LOAD_MODEL:
        dataset_parameters = DatasetParameters(
            batch_size = default_batch_size,
            sequence_length = default_sequence_length,
            sequence_steps = default_sequence_steps,
            prediction_steps = default_prediction_steps,
            fill_nan = fill_nan,
            train_start = train_start,
            train_end = train_end,
            validation_end = validation_end,
            n_predictions = default_n_predictions,
        )
        start_epoch = 0

        model = setup_model(
            in_channels = encoder.feature_dimension,
            encoder_in_channels = encoder_in_channels,
            encoder_out_channels = encoder_out_channels,
            kernel_size = kernel_size,
            padding = padding,
            frame_size = encoder.get_out_frame_size(frame_size),
            hidden_output_network_channels = []
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loader, validation_loader = setup_data(
        DATAPATH,
        SAVEFOLDER / "Regression.pkl",
        dataset_parameters # type: ignore
    )

        # Train
    scheduler = None
    losses = train_validation_loop(
        encoder = encoder,
        decoder = decoder,
        model = model, # type: ignore
        train_loader = train_loader,
        val_loader = validation_loader,
        criterion = criterion,
        optimizer = optimizer, # type: ignore
        num_epochs = epochs,
        start_epoch = start_epoch, # type: ignore
        device = DEVICE,
        update_function = print_to_file_on_epoch,
        path = SAVEFOLDER,
        save_n_epochs = save_every,
        dataset_parameters = dataset_parameters, # type: ignore
        teacher_forcing_ratios = teacher_forcing_ratio_list,
        losses = None,
        scheduler = scheduler,
        n_sequences = n_sequences,
        encoder_filename = auto_encoder_name
    )

if __name__ == '__main__':
    train_model()