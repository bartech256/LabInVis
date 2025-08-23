"""
Responsibility:
- Compute regression evaluation metrics.
- Currently supports:
  - MAE.
  - RMSE.
"""


import torch
import numpy as np

class Evaluator:
    def compute(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        mae = np.mean(np.abs(preds_np - targets_np))
        rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
        return {"MAE": mae, "RMSE": rmse}
