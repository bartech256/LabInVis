"""
CatBoost House Price Prediction with Hyperparameter Tuning
Integrates with existing GNN data processing pipeline
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import joblib

# Project-specific imports
from config import Config
from data_processor import DataProcessor
from evaluator import Evaluator


class CatBoostExperimentRunner:
    """
    Runner for CatBoost experiments with hyperparameter tuning.
    Handles data preparation, model training, evaluation, and saving results.
    """

    def __init__(self):
        """Initialize experiment runner with configuration and data processor."""
        self.cfg = Config()
        self.processor = DataProcessor(self.cfg)
        self.best_model = None
        self.best_params = None
        self.study = None

    def prepare_data(self):
        """
        Load, process, and split data using the existing DataProcessor.

        Returns:
            tuple: ((train_X, train_y), (val_X, val_y), (test_X, test_y))
        """
        print("Loading and processing data...")
        df = self.processor.load_raw()
        df = self.processor.engineer_features(df)
        df = self.processor.neighborhood_features(df)

        X, y = self.processor.preprocess(df)
        return self.processor.train_val_test_split(X, y)

    def objective(self, trial, train_data, val_data):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object
            train_data (tuple): (train_X, train_y)
            val_data (tuple): (val_X, val_y)

        Returns:
            float: RMSE on validation set
        """
        train_X, train_y = train_data
        val_X, val_y = val_data

        # Define search space
        params = self._get_hyperparameter_space(trial)

        try:
            model = CatBoostRegressor(**params)
            model.fit(train_X, train_y, eval_set=(val_X, val_y), verbose=False)

            val_pred = model.predict(val_X)
            rmse = np.sqrt(mean_squared_error(val_y, val_pred))

            if np.isnan(rmse) or np.isinf(rmse):
                return float('inf')
            return rmse

        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')

    @staticmethod
    def _get_hyperparameter_space(trial):
        """
        Define the Optuna hyperparameter search space for CatBoost.

        Args:
            trial (optuna.trial.Trial): Optuna trial object

        Returns:
            dict: CatBoost hyperparameters
        """
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 128),
            "random_strength": trial.suggest_float("random_strength", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "verbose": False,
            "random_seed": 42,
            "thread_count": -1,
            "eval_metric": "RMSE",
            "od_wait": 100,
            "use_best_model": True,
        }

        # Bootstrap type dependent params
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
        params["bootstrap_type"] = bootstrap_type
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 5)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.7, 1.0)

        return params

    def tune_hyperparameters(self, train_data, val_data, n_trials=100):
        """
        Run Optuna hyperparameter tuning.

        Args:
            train_data (tuple): (train_X, train_y)
            val_data (tuple): (val_X, val_y)
            n_trials (int): Number of Optuna trials

        Returns:
            dict: Best hyperparameters
        """
        print(f"Starting hyperparameter tuning with {n_trials} trials...")

        self.study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            study_name="catboost_optimization"
        )

        self.study.optimize(
            lambda trial: self.objective(trial, train_data, val_data),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        print(f"Best RMSE: {self.study.best_value:.8f}")
        print(f"Best parameters: {self.best_params}")
        return self.best_params

    def train_best_model(self, train_data, val_data, best_params):
        """
        Train CatBoost with the best hyperparameters.

        Args:
            train_data (tuple): (train_X, train_y)
            val_data (tuple): (val_X, val_y)
            best_params (dict): Best hyperparameters

        Returns:
            CatBoostRegressor: Trained model
        """
        print("Training final model with best parameters...")
        train_X, train_y = train_data
        val_X, val_y = val_data

        # Clean and finalize parameters
        params = self._clean_params(best_params)
        print(f"Final model parameters: {params}")

        model = CatBoostRegressor(**params)
        model.fit(
            train_X,
            train_y,
            eval_set=(val_X, val_y),
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=100
        )
        self.best_model = model
        return model

    @staticmethod
    def _clean_params(params):
        """
        Remove incompatible or unnecessary parameters.

        Args:
            params (dict): Original parameter dictionary

        Returns:
            dict: Cleaned parameters
        """
        clean_params = params.copy()
        # Remove parameters that may conflict
        for key in ["od_wait", "od_type", "use_best_model"]:
            clean_params.pop(key, None)

        # Add required fixed parameters
        clean_params.update({
            "verbose": False,
            "random_seed": 42,
            "thread_count": -1,
            "eval_metric": "RMSE",
        })

        # Handle bootstrap-specific params
        if clean_params.get("bootstrap_type") == "Bernoulli":
            clean_params.pop("bagging_temperature", None)
        elif clean_params.get("bootstrap_type") == "Bayesian":
            clean_params.pop("subsample", None)

        return clean_params

    def evaluate_model(self, model, val_data, test_data):
        """
        Evaluate model on validation and test sets.

        Args:
            model (CatBoostRegressor): Trained model
            val_data (tuple): (val_X, val_y)
            test_data (tuple): (test_X, test_y)

        Returns:
            tuple: (metrics dict, val_predictions, test_predictions)
        """
        print("Evaluating model...")

        # Evaluator wrapper to work with numpy arrays
        class EvaluatorWrapper(Evaluator):
            def compute_from_arrays(self, preds, targets):
                import torch
                preds_tensor = torch.tensor(preds, dtype=torch.float32)
                targets_tensor = torch.tensor(targets, dtype=torch.float32)
                return self.compute(preds_tensor, targets_tensor)

        evaluator = EvaluatorWrapper(label_scaler=self.processor.label_scaler)

        val_X, val_y = val_data
        test_X, test_y = test_data

        val_preds = model.predict(val_X)
        test_preds = model.predict(test_X)

        metrics = {
            "Validation": evaluator.compute_from_arrays(val_preds, val_y),
            "Test": evaluator.compute_from_arrays(test_preds, test_y),
            "best_validation_rmse": float(np.sqrt(mean_squared_error(val_y, val_preds))),
        }

        return metrics, val_preds, test_preds

    def save_results(self, model, metrics, val_preds, test_preds, exp_path):
        """
        Save model, metrics, predictions, and optimization results.

        Args:
            model (CatBoostRegressor): Trained model
            metrics (dict): Evaluation metrics
            val_preds (np.ndarray): Validation predictions
            test_preds (np.ndarray): Test predictions
            exp_path (str): Path to save experiment outputs
        """
        print("Saving results...")
        os.makedirs(exp_path, exist_ok=True)

        # Save model
        model.save_model(os.path.join(exp_path, "catboost_model.cbm"))
        joblib.dump(model, os.path.join(exp_path, "catboost_model.pkl"))

        # Save metrics
        with open(os.path.join(exp_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4, default=str)

        # Save best parameters
        if self.best_params:
            with open(os.path.join(exp_path, "best_params.json"), "w") as f:
                json.dump(self.best_params, f, indent=4)

        # Save predictions
        pd.DataFrame({"true": val_preds, "pred": val_preds}).to_csv(
            os.path.join(exp_path, "val_predictions.csv"), index=False
        )
        pd.DataFrame({"true": test_preds, "pred": test_preds}).to_csv(
            os.path.join(exp_path, "test_predictions.csv"), index=False
        )

    def run_experiment(self, n_trials=100):
        """
        Complete workflow: data prep, hyperparameter tuning, training, evaluation, and saving.

        Args:
            n_trials (int): Number of Optuna trials

        Returns:
            tuple: (experiment path, metrics)
        """
        experiment_name = getattr(self.cfg, "experiment_name", "catboost_experiment")
        print(f"Running CatBoost experiment: {experiment_name}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_path = os.path.join(
            getattr(self.cfg, "save_experiment_path", "experiments/"),
            f"catboost_exp_{timestamp}"
        )

        train_data, val_data, test_data = self.prepare_data()
        best_params = self.tune_hyperparameters(train_data, val_data, n_trials)
        model = self.train_best_model(train_data, val_data, best_params)
        metrics, val_preds, test_preds = self.evaluate_model(model, val_data, test_data)
        self.save_results(model, metrics, val_preds, test_preds, exp_path)

        print(f"Experiment completed and saved to: {exp_path}")
        return exp_path, metrics


def main():
    """Entry point to run CatBoost experiment"""
    runner = CatBoostExperimentRunner()
    exp_path, metrics = runner.run_experiment(n_trials=100)

    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Experiment saved to: {exp_path}")
    print("Test Metrics (Scaled):")
    for metric, value in metrics["Test"]["scaled"].items():
        print(f"  {metric}: {value:.8f}")

    if metrics["Test"]["real"] is not None:
        print("\nTest Metrics (Real Scale):")
        for metric, value in metrics["Test"]["real"].items():
            print(f"  {metric}: {value:.8f}")


if __name__ == "__main__":
    main()
