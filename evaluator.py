"""
Responsibility:
- Compute regression evaluation metrics.
- Currently supports:
  - MAE.
  - RMSE.
"""

import torch
import numpy as np
import pickle
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata
from sklearn.linear_model import LinearRegression


class Evaluator:
    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        mae = np.mean(np.abs(preds_np - targets_np))
        mse = np.mean((preds_np - targets_np) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((targets_np - preds_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    def msdiff(self, preds_with: torch.Tensor, preds_without: torch.Tensor) -> float:
        """Compute MSdiff: mean squared difference between two prediction settings"""
        p1 = preds_with.detach().cpu().numpy()
        p2 = preds_without.detach().cpu().numpy()
        return np.mean((p1 - p2) ** 2)

    def save_results(self, results: Dict[str, float], filename: str = "results.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(results, f)

    def load_results(self, filename: str = "results.pkl") -> Dict[str, float]:
        with open(filename, "rb") as f:
            return pickle.load(f)

    # ------------------ Baseline Model ------------------ #
    def linear_regression(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray = None):
        """
        Train and evaluate Linear Regression as baseline.
        - X, y: training data
        - X_test: optional test set
        Returns fitted model + predictions.
        """
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X if X_test is None else X_test)
        return model, preds

    # ------------------ Statistical Tests ------------------ #
    def friedman_test(self, *models: List[List[float]]):
        """Friedman test across multiple models/datasets"""
        return friedmanchisquare(*models)

    def holm_test(self, scores: np.ndarray, alpha: float = 0.05):
        """
        Holm step-down test.
        scores: 2D array (n_datasets x n_models), lower = better.
        """
        n_datasets, n_models = scores.shape
        ranks = np.array([rankdata(row).tolist() for row in scores])
        avg_ranks = np.mean(ranks, axis=0)

        comparisons = []
        model_indices = list(range(n_models))
        base = np.argmin(avg_ranks)  # best model

        for m in model_indices:
            if m == base:
                continue
            diff = abs(avg_ranks[m] - avg_ranks[base])
            # simplified p-value approx
            p_val = 2 * (1 - (diff / np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))))
            comparisons.append((base, m, p_val))

        comparisons.sort(key=lambda x: x[2])  # sort by p-value
        m = len(comparisons)
        results = []
        for i, (b, m_idx, p) in enumerate(comparisons):
            adj_alpha = alpha / (m - i)
            results.append({"compare": (b, m_idx), "p": p, "adj_alpha": adj_alpha, "reject": p < adj_alpha})
        return results

    # ------------------ Visualization ------------------ #
    def plot_scatter(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.detach().cpu().numpy()
        t = targets.detach().cpu().numpy()
        plt.scatter(t, p, alpha=0.6)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title("Predicted vs True")
        plt.show()

    def plot_error_hist(self, preds: torch.Tensor, targets: torch.Tensor):
        errors = preds.detach().cpu().numpy() - targets.detach().cpu().numpy()
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel("Prediction error")
        plt.ylabel("Frequency")
        plt.title("Error distribution")
        plt.show()

    def plot_metrics_boxplot(self, metrics: Dict[str, List[float]]):
        plt.boxplot(metrics.values(), labels=metrics.keys())
        plt.title("Metric distributions across models/datasets")
        plt.show()
