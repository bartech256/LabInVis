"""
Responsibility:
- Orchestrate the entire experiment pipeline:
  1. Load config
  2. Load processed data
  3. Build graph (if required)
  4. Create model
  5. Train model using Trainer
  6. Evaluate results with Evaluator
  7. Visualize results with Visualizer
  8. Save outputs to experiments/exp_xxx/
"""

import os
from datetime import datetime
import torch
from config import Config
from data_processor import DataProcessor
from graph_builder import GraphBuilder
from model_factory import ModelFactory
from trainer import Trainer
from evaluator import Evaluator
from utils import print_hardware_info, save_config, save_metrics

class ExperimentRunner:
    def __init__(self, config_path: str):
        self.cfg = Config.from_file(config_path)
        self.data_processor = DataProcessor(self.cfg)
        self.model_factory = ModelFactory()
        self.evaluator = Evaluator()

    def run_experiment_list(self):
        for model_name in self.cfg.models_to_run:
            self.run_experiment(model_name)

    def run_experiment(self, model_name):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_path = os.path.join(self.cfg.save_path, f"{model_name}_{timestamp}")
        os.makedirs(exp_path, exist_ok=True)

        (X_train,y_train), (X_val,y_val), (X_test,y_test) = self.data_processor.load_or_create_data()

        if "GNN" in model_name:
            train_graph = GraphBuilder(X_train, None, y_train, self.cfg).build()
            val_graph   = GraphBuilder(X_val, None, y_val, self.cfg).build()
        else:
            train_graph, val_graph = None, None

        model = self.model_factory.create_model(self.cfg)
        trainer = Trainer(model, self.cfg)
        trainer.fit([train_graph],[val_graph])

        preds = trainer.predict([val_graph])
        metrics = self.evaluator.compute(torch.tensor(preds), torch.tensor(y_val))

        torch.save(model.state_dict(), os.path.join(exp_path,"model.pt"))
        save_config(self.cfg, os.path.join(exp_path,"config.yaml"))
        save_metrics(metrics, os.path.join(exp_path,"metrics.json"))
