import numpy as np
import torch
from datetime import datetime
from typing import Callable, List, Tuple
from DatasetOperations import Loss, DatasetParameters, SLADataset
from torch.utils.data import DataLoader
from pathlib import Path
from SaveLoadModels import save_checkpoint, SaveModel
import random

try:
    from ..Models import _types
except:
    from Models import _types

loss_function = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
def predict(model: SaveModel, loader: DataLoader[SLADataset], frame_size: Tuple[int, int], device: torch.device) -> Tuple[_types.float_like, _types.float_like, _types.float_like]:
    """Makes a prediction for each batch in the loader an saves it in a tensor"""
    dataset: SLADataset = loader.dataset #type: ignore
    result = np.full((len(dataset), *frame_size), np.nan, dtype=np.float64)
    targets = np.zeros((len(dataset), *frame_size), dtype=np.float64)
    target_times = np.zeros(len(dataset))
    model.eval()
    current_idx = 0
    with torch.no_grad():
        for features, target, mask, _, result_time in loader:
            # Save time
            batch_size = result_time.size(0)
            target_times[current_idx:current_idx + batch_size] = result_time.numpy()
            
            if mask.all():
                current_idx += batch_size
                continue
            # Move to device
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # Predict
            output = model.forward_state(features).squeeze(1)
            
            # Mask array
            output[mask] = np.nan
            target[mask] = np.nan

            # Save predictions and targets
            result[current_idx:current_idx + batch_size] = output.cpu().detach().numpy()
            targets[current_idx:current_idx + batch_size] = target.cpu().detach().numpy()
            
            current_idx += batch_size
    return result, targets, target_times


def predict_seq(model: SaveModel, loader: DataLoader[SLADataset], frame_size: Tuple[int, int], device: torch.device, n_sequence: int) -> Tuple[_types.float_like, _types.float_like, _types.float_like]:
    """Makes a prediction for each batch in the loader an saves it in a tensor"""
    dataset: SLADataset = loader.dataset #type: ignore
    result = np.full((len(dataset), *frame_size), np.nan, dtype=np.float64)
    targets = np.zeros((len(dataset), *frame_size), dtype=np.float64)
    target_times = np.zeros(len(dataset))
    model.eval()
    current_idx = 0
    with torch.no_grad():
        for features, target, mask, _, result_time in loader:
            # Save time
            batch_size = result_time.size(0)
            target_times[current_idx:current_idx + batch_size] = result_time[:, n_sequence - 1].numpy()
            
            if mask.all():
                current_idx += batch_size
                continue
            # Move to device
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # Predict
            output = model.forward_state(features).squeeze(1)
            
            # Mask array
            output[mask] = np.nan
            target[mask] = np.nan

            # Save predictions and targets
            result[current_idx:current_idx + batch_size] = output[:, n_sequence - 1].cpu().detach().numpy()
            targets[current_idx:current_idx + batch_size] = target[:, n_sequence - 1].cpu().detach().numpy()
            
            current_idx += batch_size
    return result, targets, target_times

def validate(model: SaveModel, loader: DataLoader[SLADataset], criterion: loss_function, device: torch.device) -> float:
    """Computes the criterion for the model and loader in evaluation mode"""
    val_loss: float = 0
    model.eval()
    with torch.no_grad():
        for features, target, mask, _, _ in loader:
            if mask.all():
                continue
            # Convert to correct device
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # Predict
            output = model.forward_state(features)
            
            # Compute and add loss
            val_loss += criterion(output.squeeze(1), target, mask).item()

    # Compute average loss
    return val_loss / len(loader)

def train(
    model: SaveModel,
    loader: DataLoader[SLADataset],
    criterion: loss_function,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float
) -> float:
    """Traines the model based on the criterion and loader"""
    
    train_loss = 0
    model.train()
    for i, (features, target, mask, _, _) in enumerate(loader):
        # Check if day is nan
        if mask.all():
            continue
        
        # Convert to correct device
        features = features.to(device)
        target = target.to(device)
        mask = mask.to(device)
        
        # Check teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        # Predict
        if use_teacher_forcing:
            output = model.forward_state(features)
        else:
            output = model.forward_state(features, target)
        
        # Compute loss and apply backpropagation
        loss = criterion(output.squeeze(1), target, mask)
        loss.backward()
        
        # Optimize
        optimizer.step()
        optimizer.zero_grad()
        
        # Save loss
        train_loss += loss.item()
        if np.isnan(train_loss):
            raise ValueError(f"Found nan values at {i}. Please use other hyperparameters or try again. Loss was {loss}. use_teacher_forcing = {use_teacher_forcing}. Output was nan? {np.isnan(output.detach().cpu().numpy()).all()}. Input was nan? {np.isnan(features.detach().cpu().numpy()).all()}. Targets was nan? {np.isnan(target.detach().cpu().numpy()).all()}")

    # Compute average loss
    return train_loss / len(loader)

def train_validation_loop(
    model: SaveModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: loss_function,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    start_epoch: int,
    device: torch.device,
    update_function: Callable[[Loss], None] | None = None,
    path: Path | None = None,
    save_n_epochs: int | None = None,
    dataset_parameters: DatasetParameters | None = None,
    teacher_forcing_ratio: List[float] | float = 0,
    losses: List[Loss] | None = None,
    scheduler: torch.optim.lr_scheduler.Optimizer | None = None
) -> List[Loss]:
    """ Trains and evaluates the model at each epoch"""
    learning_rate = 0
    
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
    
    if losses is None:
        losses = []
    
    try:
        for epoch in range(start_epoch + 1, num_epochs + 1):
            if isinstance(teacher_forcing_ratio, (tuple, list)):
                current_teacher_forcing_ratio = teacher_forcing_ratio[epoch - 1]
            else:
                current_teacher_forcing_ratio = teacher_forcing_ratio
            
            # Train model
            train_loss = train(model, train_loader, criterion, optimizer, device, current_teacher_forcing_ratio)

            # Compute validation loss
            val_loss = validate(model, val_loader, criterion, device)

            # Save losses
            if scheduler is not None:
                learning_rate = scheduler.get_last_lr()
                
            losses.append(Loss(epoch, train_loss, val_loss, datetime.now(), learning_rate))

            # Update scheduler
            if scheduler is not None:
                scheduler.step(epoch)
            
            # Print
            if update_function is not None:
                update_function(losses[-1])

            # Save checkpoint of model
            if save_n_epochs is not None and path is not None and dataset_parameters is not None:
                if epoch % save_n_epochs == 0:
                    save_checkpoint(path / f"checkpoint_{epoch}.pkl", model, optimizer, losses[-1], dataset_parameters)
    except KeyboardInterrupt:
        pass
    return losses