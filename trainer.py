"""
Responsibility:
- Handle the training and validation loop for models.
- Save the best model based on validation performance.
- Provide prediction method for trained models.
"""


import torch
import numpy as np
from tqdm import tqdm

class Trainer:
    """
    Handles the training, validation, and prediction loop for models.
    """
    def __init__(self, model, optimizer, criterion, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.best_val_loss = float("inf")
        self.device = torch.device(cfg.default_device)
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train_epoch(self, train_data):
        self.model.train()
        total_loss = 0
        for data in train_data:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(data)
            loss = self.criterion(preds, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_data)

    def validate(self, val_data):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for data in val_data:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, data.y)
                total_loss += loss.item()
                preds.append(out.cpu().numpy())
                targets.append(data.y.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        return total_loss / len(val_data), mae, rmse

    def fit(self, train_data, val_data):
        epoch_pbar = tqdm(range(self.cfg.max_epochs), desc="Training")
        for epoch in epoch_pbar:
            train_loss = self.train_epoch(train_data)
            val_loss, mae, rmse = self.validate(val_data)
            epoch_pbar.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}"
            })
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
        print(f"✅ Training complete. Best Val Loss: {self.best_val_loss:.4f}")

    def predict(self, data_list):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)
                predictions.append(self.model(data).cpu().numpy())
        return np.concatenate(predictions, axis=0)
