import numpy as np
import torch
from torch.optim import lr_scheduler
from datetime import datetime
from typing import Callable, List, Tuple, Sequence
from torch.utils.data import DataLoader
from pathlib import Path
from AltimeterAutoencoder.src import _types
from .SaveLoadModels import save_checkpoint, SaveModel
from ..Shared import Loss, DatasetParameters, SLADataset

loss_function = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def predict_seq(model, loader: DataLoader[SLADataset], frame_size: Tuple[int, int], device: torch.device, n_sequences: int) -> Tuple[_types.float_like, _types.float_like, _types.float_like]:
    """Makes a prediction for each batch in the loader an saves it in a tensor"""
    dataset: SLADataset = loader.dataset #type: ignore
    result = np.full((len(dataset), n_sequences, *frame_size), np.nan, dtype=np.float64)
    targets = np.zeros((len(dataset), n_sequences, *frame_size), dtype=np.float64)
    target_times = np.zeros((len(dataset), n_sequences))
    model.eval()
    current_idx = 0
    with torch.no_grad():
        for features, target, mask, _, result_time in loader:            
            # Save time
            batch_size = result_time.size(0)
            target_times[current_idx:current_idx + batch_size] = result_time.numpy()
            
            if torch.any(torch.all(torch.all(mask, dim=3), dim=2)):
                current_idx += batch_size
                continue
                            
            # Move to device
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # Predict
            output = model(features, None, n_sequences, 0)
           
            # Mask array
            output = output
            target = target.squeeze(1)
            output[mask] = np.nan
            target[mask] = np.nan

            # Save predictions and targets
            result[current_idx:current_idx + batch_size] = output.cpu().detach().numpy()
            targets[current_idx:current_idx + batch_size] = target.cpu().detach().numpy()
            
            current_idx += batch_size
    return result, targets, target_times

def validate(
    model: SaveModel,
    loader: DataLoader[SLADataset], 
    criterion: loss_function,
    device: torch.device,
    n_sequences: int | None
) -> Tuple[float, _types.float_like, _types.float_like]:
    """Computes the criterion for the model and loader in evaluation mode"""
    val_loss: float = 0
    example_img_true = np.array(0, dtype=np.float32)
    example_img_pred = np.array(0, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for features, target, mask, _, _ in loader:            
            if torch.any(torch.all(torch.all(mask, dim=3), dim=2)):
                continue
                       
            # Convert to correct device
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # Predict
            output = model(features, target, n_sequences, 0)
            
            # Compute and add loss
            if n_sequences is None:
                current_loss = criterion(output, target, mask)
            else:
                current_loss = criterion(output[:, :n_sequences], target[:, :n_sequences], mask[:, :n_sequences])
            current_loss = current_loss.item()
            val_loss += current_loss
            
            if example_img_true is None:
                example_img_pred = output[0].cpu().numpy()
                example_img_true = target[0].cpu().numpy()
                m = mask[0, :n_sequences].cpu().numpy()
                example_img_pred[m] = np.nan
                example_img_true[m] = np.nan

    # Compute average loss
    return val_loss / len(loader), example_img_true, example_img_pred

def train(
    model: SaveModel,
    loader: DataLoader[SLADataset],
    criterion: loss_function,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_sequence: int,
    teacher_forcing_ratio: float | Sequence[float] = 0,
) -> float:
    """Traines the model based on the criterion and loader"""
    
    train_loss = 0
    model.train()
    for i, (features, target, mask, _, _) in enumerate(loader):       
        # Check if day is nan
        if torch.any(torch.all(torch.all(mask, dim=3), dim=2)):
            continue
                
        # Convert to correct device
        features = features.to(device)
        target = target.to(device)
        mask = mask.to(device)
        
        # Predict
        output = model(features, target, n_sequence, teacher_forcing_ratio)
        
        # Compute loss and apply backpropagation
        loss = criterion(output[:, :n_sequence], target[:, :n_sequence], mask[:, :n_sequence])
        loss.backward()
        
        # Optimize
        optimizer.step()
        optimizer.zero_grad()
        
        # Save loss
        train_loss += loss.item()
        
        if np.isnan(train_loss):
            raise ValueError(f"Found nan values at {i}. Please use other hyperparameters or try again. Loss was {loss}. Output was nan? {np.isnan(output.detach().cpu().numpy()).any()}. Input was nan? {np.isnan(features.detach().cpu().numpy()).all()}. Targets was nan? {np.isnan(target.detach().cpu().numpy()).any()}")

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
    n_sequences: int,
    update_function: Callable[[Loss, _types.float_like, _types.float_like, int], None] | None = None,
    path: Path | None = None,
    save_n_epochs: int | None = None,
    dataset_parameters: DatasetParameters | None = None,
    teacher_forcing_ratios: List[float] | float = 0,
    losses: List[Loss] | None = None,
    scheduler: lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau | None = None,
) -> List[Loss]:
    """ Trains and evaluates the model at each epoch"""
    teacher_forcing_ratio = teacher_forcing_ratios
    
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
    
    if losses is None:
        losses = []
    
    try:
        for epoch in range(start_epoch + 1, num_epochs + 1):
            if isinstance(teacher_forcing_ratios, Sequence):
                teacher_forcing_ratio = teacher_forcing_ratios[epoch]
            # Train model
            train_loss = train(model, train_loader, criterion, optimizer, device, n_sequences, teacher_forcing_ratio)

            # Compute validation loss
            val_loss, example_img_true, example_img_pred = validate(model, val_loader, criterion, device, n_sequences)

            # Save losses
            learning_rate = optimizer.param_groups[0]['lr']    
            losses.append(Loss(epoch, train_loss, val_loss, datetime.now(), learning_rate))

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.LRScheduler):
                    scheduler.step(epoch)
                else:
                    scheduler.step(val_loss)
            
            # Print
            if update_function is not None:
                update_function(losses[-1], example_img_true, example_img_pred, num_epochs)

            # Save checkpoint of model
            if save_n_epochs is not None and path is not None and dataset_parameters is not None:
                if epoch % save_n_epochs == 0:
                    save_checkpoint(path / f"checkpoint_{epoch}.pkl", model, optimizer, losses[-1], dataset_parameters)
            
    except KeyboardInterrupt:
        pass
    return losses