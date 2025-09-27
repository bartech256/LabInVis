"""
Responsibility:
- Handle the training and validation loop for models.
- Save the best model based on validation performance.
- Provide prediction method for trained models.
"""

import torch
import numpy as np
from tqdm import tqdm
from evaluator import Evaluator   

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

        self.evaluator = Evaluator()
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []  # list of dicts with all metrics (scaled + real)

    def train_epoch(self, train_data):
        """One epoch of training"""
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
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for data in val_data:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, data.y)
                total_loss += loss.item()
                preds.append(out.cpu())
                targets.append(data.y.cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        metrics = self.evaluator.compute(preds, targets)
        return total_loss / len(val_data), metrics

    def fit(self, train_data, val_data):
        """Full training loop with history tracking"""
        epoch_pbar = tqdm(range(self.cfg.max_epochs), desc="Training")
        for epoch in epoch_pbar:
            train_loss = self.train_epoch(train_data)
            val_loss, metrics = self.validate(val_data)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(metrics)

            # Display scaled metrics in progress bar
            scaled_metrics = metrics.get("scaled", {})
            display_metrics = {k: f"{v:.4f}" for k, v in scaled_metrics.items()}

            epoch_pbar.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                **display_metrics
            })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")

        print(f"Training complete. Best Val Loss: {self.best_val_loss:.4f}")

    def predict(self, data_list):
        """Run inference on new data"""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)
                predictions.append(self.model(data).cpu().numpy())
        return np.concatenate(predictions, axis=0)
