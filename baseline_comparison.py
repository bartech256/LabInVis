"""
CatBoost House Price Prediction with Hyperparameter Tuning
Reuses data processing pipeline from existing GNN infrastructure
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import joblib

# Import from existing infrastructure
from config import Config
from data_processor import DataProcessor
from evaluator import Evaluator

class CatBoostExperimentRunner:
    def __init__(self):
        """Initialize CatBoost experiment runner"""
        self.cfg = Config()  # Use default config
        self.processor = DataProcessor(self.cfg)
        self.best_model = None
        self.best_params = None
        self.study = None

    def prepare_data(self):
        """Prepare data using existing data processor"""
        print("Loading and processing data...")
        df = self.processor.load_raw()
        df = self.processor.engineer_features(df)
        df = self.processor.neighborhood_features(df)

        X, y = self.processor.preprocess(df)
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = self.processor.train_val_test_split(X, y)

        return (train_X, train_y), (val_X, val_y), (test_X, test_y)

    def objective(self, trial, train_data, val_data):
        """Objective function for Optuna hyperparameter optimization"""
        train_X, train_y = train_data
        val_X, val_y = val_data

        # Define base hyperparameter search space
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'random_strength': trial.suggest_float('random_strength', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
            'verbose': False,
            'random_seed': 42,
            'thread_count': -1,
            'eval_metric': 'RMSE',
            'od_wait': 100,
            'use_best_model': True
        }

        # Bootstrap type specific parameters
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        params['bootstrap_type'] = bootstrap_type

        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 5)
        elif bootstrap_type == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.7, 1.0)

        try:
            model = CatBoostRegressor(**params)
            model.fit(
                train_X, train_y,
                eval_set=(val_X, val_y),
                verbose=False
            )

            val_pred = model.predict(val_X)
            rmse = np.sqrt(mean_squared_error(val_y, val_pred))

            # Check for invalid results
            if np.isnan(rmse) or np.isinf(rmse):
                return float('inf')

            return rmse

        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')

    def tune_hyperparameters(self, train_data, val_data, n_trials=100):
        """Perform hyperparameter tuning using Optuna"""
        print(f"Starting hyperparameter tuning with {n_trials} trials...")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='catboost_optimization'
        )

        study.optimize(
            lambda trial: self.objective(trial, train_data, val_data),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self.study = study
        self.best_params = study.best_params

        print(f"Best RMSE: {study.best_value:.8f}")
        print(f"Best parameters: {study.best_params}")

        return study.best_params

    def train_best_model(self, train_data, val_data, best_params):
        """Train the final model with best parameters"""
        print("Training final model with best parameters...")

        train_X, train_y = train_data
        val_X, val_y = val_data

        # Clean up parameters - ensure bootstrap-specific params are handled correctly
        clean_params = best_params.copy()

        # Remove any conflicting or unwanted parameters
        params_to_remove = ['od_wait', 'od_type', 'use_best_model']
        for param in params_to_remove:
            clean_params.pop(param, None)

        # Add required parameters
        clean_params.update({
            'verbose': False,
            'random_seed': 42,
            'thread_count': -1,
            'eval_metric': 'RMSE'
        })

        # Remove incompatible parameter combinations
        if clean_params.get('bootstrap_type') == 'Bernoulli':
            # Remove bagging_temperature if it exists (only for Bayesian)
            clean_params.pop('bagging_temperature', None)
        elif clean_params.get('bootstrap_type') == 'Bayesian':
            # Remove subsample if it exists (only for Bernoulli)
            clean_params.pop('subsample', None)

        print(f"Final model parameters: {clean_params}")

        model = CatBoostRegressor(**clean_params)
        model.fit(
            train_X, train_y,
            eval_set=(val_X, val_y),
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=100
        )

        self.best_model = model
        return model

    def evaluate_model(self, model, val_data, test_data):
        """Evaluate model performance using existing evaluator"""
        print("Evaluating model...")

        # Add compute_from_arrays method to work with numpy arrays
        class EvaluatorWrapper(Evaluator):
            def compute_from_arrays(self, preds, targets):
                import torch
                preds_tensor = torch.tensor(preds, dtype=torch.float32)
                targets_tensor = torch.tensor(targets, dtype=torch.float32)
                return self.compute(preds_tensor, targets_tensor)

        evaluator = EvaluatorWrapper(label_scaler=self.processor.label_scaler)

        val_X, val_y = val_data
        test_X, test_y = test_data

        # Validation predictions
        val_preds = model.predict(val_X)
        val_metrics = evaluator.compute_from_arrays(val_preds, val_y)

        # Test predictions
        test_preds = model.predict(test_X)
        test_metrics = evaluator.compute_from_arrays(test_preds, test_y)

        metrics = {
            "Validation": val_metrics,
            "Test": test_metrics,
            "best_validation_rmse": float(np.sqrt(mean_squared_error(val_y, val_preds)))
        }

        return metrics, val_preds, test_preds

    def save_optimization_plots(self, study, exp_path):
        """Save Optuna optimization plots"""
        try:
            import matplotlib.pyplot as plt
            import optuna.visualization.matplotlib as vis

            # Optimization history
            try:
                ax = vis.plot_optimization_history(study)
                fig = ax.figure
                fig.savefig(os.path.join(exp_path, "optimization_history.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Could not generate optimization history plot: {e}")

            # Parameter importances
            try:
                ax = vis.plot_param_importances(study)
                fig = ax.figure
                fig.savefig(os.path.join(exp_path, "param_importances.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Could not generate parameter importance plot: {e}")

            # Parallel coordinate plot
            try:
                ax = vis.plot_parallel_coordinate(study)
                fig = ax.figure
                fig.savefig(os.path.join(exp_path, "parallel_coordinate.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Could not generate parallel coordinate plot: {e}")

        except ImportError:
            print("Matplotlib not available for plotting optimization results")

    def save_feature_importance(self, model, exp_path):
        """Save feature importance plot"""
        try:
            import matplotlib.pyplot as plt

            feature_names = (self.cfg.embedding_features +
                           self.cfg.engineered_features if hasattr(self.cfg, 'engineered_features')
                           else [f'feature_{i}' for i in range(len(model.feature_importances_))])

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title('CatBoost Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(exp_path, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Save as CSV
            importance_df.to_csv(os.path.join(exp_path, "feature_importance.csv"), index=False)

        except ImportError:
            print("Matplotlib not available for feature importance plot")

    def run_experiment(self, n_trials=100):
        """Run complete CatBoost experiment with hyperparameter tuning"""
        experiment_name = getattr(self.cfg, 'experiment_name', 'catboost_experiment')
        print(f"Running CatBoost experiment: {experiment_name}")

        # Create output folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_path = os.path.join(
            getattr(self.cfg, "save_experiment_path", "experiments/"),
            f"catboost_exp_{timestamp}"
        )
        os.makedirs(exp_path, exist_ok=True)

        # Prepare data
        train_data, val_data, test_data = self.prepare_data()

        # Hyperparameter tuning
        best_params = self.tune_hyperparameters(train_data, val_data, n_trials)

        # Train final model
        model = self.train_best_model(train_data, val_data, best_params)

        # Evaluate model
        metrics, val_preds, test_preds = self.evaluate_model(model, val_data, test_data)

        # Save results
        print("Saving results...")

        # Save model
        model_path = os.path.join(exp_path, "catboost_model.cbm")
        model.save_model(model_path)

        # Save model with joblib as backup
        joblib_model_path = os.path.join(exp_path, "catboost_model.pkl")
        joblib.dump(model, joblib_model_path)

        # Save metrics
        metrics_path = os.path.join(exp_path, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)

        # Save best parameters
        params_path = os.path.join(exp_path, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)

        # Save basic config info
        config_info = {
            'experiment_name': 'catboost_house_prices',
            'timestamp': timestamp,
            'n_trials': n_trials,
            'data_shape': {
                'train': train_data[0].shape,
                'val': val_data[0].shape,
                'test': test_data[0].shape
            }
        }
        config_path = os.path.join(exp_path, "experiment_info.json")
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=4, default=str)

        # Save optimization results
        if self.study:
            study_path = os.path.join(exp_path, "optuna_study.pkl")
            joblib.dump(self.study, study_path)
            self.save_optimization_plots(self.study, exp_path)

        # Save feature importance
        self.save_feature_importance(model, exp_path)

        # Save predictions separately for val and test
        val_predictions_df = pd.DataFrame({
            'true': val_data[1],
            'pred': val_preds
        })
        val_predictions_df.to_csv(os.path.join(exp_path, "val_predictions.csv"), index=False)

        test_predictions_df = pd.DataFrame({
            'true': test_data[1],
            'pred': test_preds
        })
        test_predictions_df.to_csv(os.path.join(exp_path, "test_predictions.csv"), index=False)

        print(f"CatBoost experiment completed and saved to: {exp_path}")
        print(f"Final test RMSE: {metrics['Test']['scaled']['RMSE']:.8f}")
        print(f"Final test MAE: {metrics['Test']['scaled']['MAE']:.8f}")

        return exp_path, metrics

def main():
    """Main function to run CatBoost experiment"""
    runner = CatBoostExperimentRunner()

    # Run experiment with hyperparameter tuning
    # Adjust n_trials based on your computational budget (100-500 is reasonable)
    exp_path, metrics = runner.run_experiment(n_trials=100)

    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Experiment saved to: {exp_path}")
    print("\nTest Metrics (Scaled):")
    for metric, value in metrics['Test']['scaled'].items():
        print(f"  {metric}: {value:.8f}")

    if metrics['Test']['real'] is not None:
        print("\nTest Metrics (Real Scale):")
        for metric, value in metrics['Test']['real'].items():
            print(f"  {metric}: {value:.8f}")

if __name__ == "__main__":
    main()