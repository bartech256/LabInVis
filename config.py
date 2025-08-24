"""
Responsibility:
- Hold all experiment settings (hyperparameters, model type, file paths, etc.).
- Load configuration from YAML or JSON files.
"""

import torch
import os
import yaml
import json

class Config:
    def __init__(self, cfg_dict=None):
        # Sets a constant seed so that each run is deterministic
        self.random_seed = 42
        # Paths
        self.data_path = "data/"
        self.raw_data_file_path = os.path.join(self.data_path, "raw_kc.csv")
        self.processed_data_file_path = os.path.join(self.data_path, "processed_data.csv")
        self.spatial_features_file_path = os.path.join(self.data_path, "spatial_features.csv")
        self.feature_statistics_file_path = os.path.join(self.data_path, "feature_statistics.csv")
        self.train_data_file_path = os.path.join(self.data_path, "train_data.csv")
        self.val_data_file_path = os.path.join(self.data_path, "val_data.csv")
        self.test_data_file_path = os.path.join(self.data_path, "test_data.csv")
        os.makedirs(self.data_path, exist_ok=True)

        # Spatial features
        self.geo_features = ["lat", "long"]
        # Basic features that go into the model
        self.embedding_features = [
            "bedrooms", "bathrooms", "sqft_living", "floors",
            "grade", "condition", "sqft_above", "sqft_basement"
        ]
        # New features we are creating in DataProcessor
        self.engineered_features = [
            "total_sqft", "bath_bed_ratio", "house_age", "years_since_reno",
            "has_basement", "has_been_renovated", "is_luxury", "space_efficiency",
            "neighborhood_price_mean", "neighborhood_price_median", "neighborhood_price_std"
        ]
        # Col we want to predict
        self.label_col = "price"
        # Method for creating the edges 
        self.graph_type = "radius" 
        self.radius1_k = 30
        self.radius2_k = 20
        self.radius3_k = 0
        self.top_k_sim = 7

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.max_epochs = 50
        self.criterion = "MSELoss"
        self.optimizer = "Adam"

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_device = self.device  

        # Experiment - Which models to run
        self.models_to_run = ["SimpleGCN", "SimpleGAT", "MultiLayerGCN"]
        self.save_path = "experiments/"
        os.makedirs(self.save_path, exist_ok=True)

        # Override defaults with config file values (if provided)
        if cfg_dict:
            self.__dict__.update(cfg_dict)
    # Function to load from a file
    @staticmethod
    def from_file(path: str):
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r") as f:
                return Config(yaml.safe_load(f))
        elif path.endswith(".json"):
            with open(path, "r") as f:
                return Config(json.load(f))
        else:
            raise ValueError("Config file must be YAML or JSON")
