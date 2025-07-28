import os
import yaml
import json
import torch
from datetime import datetime

from config import Config
from trainer import Trainer
from models import SimpleGCN
from data_processor import DataProcessor

def save_config(cfg: Config, path: str):
    """Save the config object as a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(cfg.__dict__, f)

def save_metrics(metrics: dict, path: str):
    """Save evaluation metrics to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def run_experiment():
    # Create a new subfolder for this experiment
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join("experiments", f"exp_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)

    # Initialize config
    cfg = Config()
    cfg.save_experiment_path = exp_path

    # Save config to YAML file
    save_config(cfg, os.path.join(exp_path, "config.yaml"))

    # Load and process data
    processor = DataProcessor(cfg)
    df = processor.load_raw()

    # Generate and save spatial features
    processor.create_spatial_features(df)

    # Preprocess features and labels
    X, y = processor.preprocess(df)

    # Create torch_geometric.data.Data object with graph
    data = processor.create_data_object(X, y)

    # Initialize model, optimizer, loss function
    model = SimpleGCN(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.default_learning_rate)
    criterion = torch.nn.MSELoss()

    # Train the model
    trainer = Trainer(model, optimizer, criterion, cfg)
    trainer.fit([data], [data])  # If using loaders in future, replace here

    # Save trained model
    model_path = os.path.join(exp_path, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save evaluation results
    metrics = {"final_val_loss": trainer.best_val_loss}
    metrics_path = os.path.join(exp_path, "metrics.json")
    save_metrics(metrics, metrics_path)

    print(f"Experiment completed and saved to: {exp_path}")

if __name__ == "__main__":
    run_experiment()
