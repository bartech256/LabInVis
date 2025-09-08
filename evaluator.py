"""
Responsibility:
- Compute regression evaluation metrics.
- Supports:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MSE (Mean Squared Error)
  - R² (Coefficient of Determination)
"""

import torch
import numpy as np

class Evaluator:
    def compute(self, preds: torch.Tensor, targets: torch.Tensor):
        # convert tensors to numpy
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Mean Absolute Error
        mae = np.mean(np.abs(preds_np - targets_np))

        # Mean Squared Error
        mse = np.mean((preds_np - targets_np) ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum((targets_np - preds_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
