import torch
from torch_geometric.data import Data
from typing import List
import numpy as np
from tqdm import tqdm

class Trainer:
    """
    Trainer class to handle the training and validation of a GNN model.
    """

    def __init__(self, model, optimizer, criterion, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.best_val_loss = float('inf')
        self.device = torch.device(cfg.default_device)
        
        # Move model to correct device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train_epoch(self, data_list: List[Data], train_masks: List[torch.Tensor] = None):
        self.model.train()
        total_loss = 0

        for i, data in enumerate(data_list):
            # Move data to correct device
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Call model with data object (model handles x and edge_index internally)
            out = self.model(data)
            
            # If masks are provided, use only relevant nodes
            if train_masks is not None and i < len(train_masks):
                mask = train_masks[i].to(self.device)
                loss = self.criterion(out[mask], data.y[mask])
            else:
                loss = self.criterion(out, data.y)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_list)

    def validate(self, data_list: List[Data], val_masks: List[torch.Tensor] = None):
        self.model.eval()
        total_loss = 0
        preds = []
        targets = []

        with torch.no_grad():
            for i, data in enumerate(data_list):
                # Move data to correct device
                data = data.to(self.device)
                
                out = self.model(data)
                
                # If masks are provided, use only relevant nodes
                if val_masks is not None and i < len(val_masks):
                    mask = val_masks[i].to(self.device)
                    loss = self.criterion(out[mask], data.y[mask])
                    preds.append(out[mask].cpu().numpy())
                    targets.append(data.y[mask].cpu().numpy())
                else:
                    loss = self.criterion(out, data.y)
                    preds.append(out.cpu().numpy())
                    targets.append(data.y.cpu().numpy())
                
                total_loss += loss.item()

        # Handle empty predictions
        if len(preds) == 0:
            return float('inf'), float('inf'), float('inf')
        
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        return total_loss / len(data_list), mae, rmse

    def fit(self, train_data: List[Data], val_data: List[Data], 
            train_masks: List[torch.Tensor] = None, val_masks: List[torch.Tensor] = None):
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(self.cfg.default_epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            train_loss = self.train_epoch(train_data, train_masks)
            val_loss, mae, rmse = self.validate(val_data, val_masks)

            # Update progress bar description
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'MAE': f'{mae:.4f}',
                'RMSE': f'{rmse:.4f}'
            })

            # Print detailed info every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                tqdm.write(f"Epoch {epoch + 1:3d}: Train Loss = {train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
        
        epoch_pbar.close()
        print(f"\n Training completed. Best validation loss: {self.best_val_loss:.4f}")

    def predict(self, data_list: List[Data]):
        """Make predictions on new data."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in tqdm(data_list, desc="Making predictions"):
                data = data.to(self.device)
                out = self.model(data)
                predictions.append(out.cpu().numpy())
        
        return np.concatenate(predictions, axis=0) if predictions else np.array([])