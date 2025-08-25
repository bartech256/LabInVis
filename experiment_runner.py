"""
Responsibility:
- Orchestrates the full experiment pipeline.
- Loads config, processes data, builds graphs, trains model, saves outputs.
"""

import os
import yaml
import json
import torch
import random
import numpy as np
from datetime import datetime
from config import Config
from graph_builder import GraphBuilder
from trainer import Trainer
from data_processor import DataProcessor
from model_factory import ModelFactory
from evaluator import Evaluator
from visualizer import Visualizer   # NEW

class ExperimentRunner:
    def __init__(self, config_path=None):
        if config_path is not None:
            with open(config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            self.cfg = Config()
            for k, v in cfg_dict.items():
                setattr(self.cfg, k, v)  # override defaults from YAML
        else:
            self.cfg = Config()

        self.model_factory = ModelFactory()
        self.processor = DataProcessor(self.cfg)

    def run_experiment_list(self):
        """Run experiments"""
        self.run_experiment(self.cfg)

    def run_experiment(self, experiment_config: Config):
        print(f"Running experiment: {getattr(experiment_config, 'experiment_name', 'unnamed')}")

        set_all_seeds(42)  # For reproducibility

        # === OUTPUT FOLDER ===
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_path = os.path.join(
            getattr(experiment_config, "save_experiment_path", "experiments/"),
            f"exp_{timestamp}"
        )
        os.makedirs(exp_path, exist_ok=True)

        # === DATA PROCESSING ===
        print("Loading and processing data...")
        df = self.processor.load_raw()
        df = self.processor.engineer_features(df)
        df = self.processor.neighborhood_features(df)

        X, y = self.processor.preprocess(df)
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = self.processor.train_val_test_split(X, y)

        geo = df[["lat", "long"]].values
        train_geo = geo[:len(train_X)]
        val_geo = geo[len(train_X):len(train_X)+len(val_X)]

        # === GRAPH BUILDING ===
        print("Building graphs")
        train_graph = GraphBuilder(train_X, train_geo, train_y, experiment_config).build()
        val_graph = GraphBuilder(val_X, val_geo, val_y, experiment_config).build()

        # === MODEL CREATION ===
        print("Creating model")
        input_dim = train_graph.x.shape[1] 
        experiment_config.GNN_model_params = {"input_dim": input_dim}

        model = self.model_factory.create_model(experiment_config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=getattr(experiment_config, "learning_rate", 0.001)
        )
        criterion = torch.nn.MSELoss()

        trainer = Trainer(model, optimizer, criterion, experiment_config)
        trainer.fit([train_graph], [val_graph])

        # === EVALUATION ===
        print("Evaluating model")
        evaluator = Evaluator()
        preds = trainer.predict([val_graph])
        targets = val_graph.y.cpu()

        metrics = evaluator.compute(torch.tensor(preds), targets)
        metrics["best_val_loss"] = float(trainer.best_val_loss)
        metrics = {k: float(v) for k, v in metrics.items()}

        # === VISUALIZATION ===
        visualizer = Visualizer()
        visualizer.plot_training_loss(trainer.train_losses, exp_path)
        visualizer.plot_validation_metrics(
            {"Val Loss": trainer.val_losses, "MAE": trainer.val_mae, "RMSE": trainer.val_rmse},
            exp_path
        )
        visualizer.plot_graph(train_graph, exp_path)

        # === SAVE RESULTS ===
        model_path = os.path.join(exp_path, "model.pt")
        self.model_factory.save_model(model, model_path)

        metrics_path = os.path.join(exp_path, "metrics.json")
        save_metrics(metrics, metrics_path)

        config_out_path = os.path.join(exp_path, "config.yaml")
        save_config(experiment_config, config_out_path)

        print(f"Experiment completed and saved to: {exp_path}")


# === HELPERS ===

def save_config(cfg: Config, path: str):
    """Save the config object as a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(cfg.__dict__, f)  # dump attributes of Config


def save_metrics(metrics: dict, path: str):
    """Save evaluation metrics to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def set_all_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)