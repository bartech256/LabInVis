"""
Responsibility:
- Evaluate regression predictions.
- Compute standard metrics on both scaled and real (inverse-transformed) data.
"""

import torch
import numpy as np


class Evaluator:
    """
    Computes regression metrics (MAE, MSE, RMSE, R2, MAPE) on scaled
    and optionally inverse-transformed data using a provided scaler.
    """

    def __init__(self, label_scaler=None):
        """
        Args:
            label_scaler: Optional scaler used to inverse-transform labels.
        """
        self.label_scaler = label_scaler

    def compute(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Compute metrics on scaled and optionally real labels.

        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): True values.

        Returns:
            dict: {
                "scaled": dict of metrics,
                "real": dict of metrics (if label_scaler is provided)
            }
        """
        preds_np = preds.detach().cpu().numpy().reshape(-1, 1)
        targets_np = targets.detach().cpu().numpy().reshape(-1, 1)

        scaled_metrics = self._compute_metrics(preds_np, targets_np)
        real_metrics = None

        if self.label_scaler is not None:
            preds_real = self.label_scaler.inverse_transform(preds_np)
            targets_real = self.label_scaler.inverse_transform(targets_np)
            real_metrics = self._compute_metrics(preds_real, targets_real)

        return {
            "scaled": scaled_metrics,
            "real": real_metrics
        }

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray):
        """
        Compute standard regression metrics.

        Args:
            preds (np.ndarray): Predicted values.
            targets (np.ndarray): True values.

        Returns:
            dict: Metrics dictionary containing MAE, MSE, RMSE, R2, MAPE.
        """
        # Mean Absolute Error
        mae = np.mean(np.abs(preds - targets))
        # Mean Squared Error
        mse = np.mean((preds - targets) ** 2)
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        # R² Score
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Mean Absolute Percentage Error (handle division by zero)
        non_zero_mask = np.abs(targets) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((targets[non_zero_mask] - preds[non_zero_mask]) /
                                   targets[non_zero_mask])) * 100
        else:
            mape = float('inf')

        return {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape)
        }
