import os
import yaml
import json
import torch
from datetime import datetime

from config import Config
from graph_builder import GraphBuilder
from trainer import Trainer
from data_processor import DataProcessor
from model_factory import ModelFactory

class ExperimentRunner:
    """A class to run experiments based on configurations."""
    def __init__(self):
        self.model_factory = ModelFactory()
        self.processor = DataProcessor()
        self.graph_builder = GraphBuilder()

    def run_experiment_list(self, configs: list):
        """Run all experiments in the provided configuration list."""
        for experiment_config in self.configs:
            print(f"Running experiment with config: {experiment_config.experiment_name}")
            self.run_experiment(experiment_config)

    def run_experiment(self, experiment_config: Config):
        # Create a new subfolder for this experiment
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_path = os.path.join("experiments", f"exp_{timestamp}")
        os.makedirs(exp_path, exist_ok=True)

        # load train data
        train_data = self.processor.load_train(experiment_config)

        # load validation data
        val_data = self.processor.load_val(experiment_config)

        # maybe load spatial features if needed

        # build graph from train data according to config
        train_graph = self.graph_builder.build_graph(train_data, experiment_config)
        val_graph = self.graph_builder.build_graph(val_data, experiment_config)  # maybe pass spatial features if needed

        # create model according to config
        model = self.model_factory.create_model(experiment_config)

        # Train the model
        trainer = Trainer(model, train_data, val_data, train_graph, val_graph, experiment_config)
        trainer.fit()

        # Save trained model
        model_path = os.path.join(exp_path, "model.pt")
        self.model_factory.save(model, model_path)

        # Save evaluation results
        metrics = {"final_val_loss": trainer.best_val_loss}
        metrics_path = os.path.join(exp_path, "metrics.json")
        save_metrics(metrics, metrics_path)

        # Save config to YAML file
        save_config(experiment_config, os.path.join(exp_path, "config.yaml"))

        print(f"Experiment completed and saved to: {exp_path}")

        # Visualize results with cfg specifies visualization





def save_config(cfg: Config, path: str):
    """Save the config object as a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(cfg.__dict__, f)

def save_metrics(metrics: dict, path: str):
    """Save evaluation metrics to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
