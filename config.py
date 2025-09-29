import os
import torch
import yaml
import json


class Config:
    """
    Configuration holder for experiments.

    Attributes:
        random_seed (int): Seed for reproducibility.
        data_path (str): Base data directory.
        geo_features (list): Geographic features.
        embedding_features (list): Basic input features.
        engineered_features (list): Features created in DataProcessor.
        label_col (str): Target column name.
        graph_type (str): Graph edge construction method.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for training.
        max_epochs (int): Maximum number of training epochs.
        criterion (str): Loss function name.
        optimizer (str): Optimizer name.
        device (str): Device to use ('cuda' or 'cpu').
        default_device (str): Default device.
        models_to_run (list): List of model names to run.
        save_path (str): Directory to save experiment outputs.
    """

    def __init__(self, cfg_dict=None):
        """
        Initialize default configuration and optionally override with a dictionary.

        Args:
            cfg_dict (dict, optional): Dictionary to override default settings.
        """
        self.random_seed = 42

        # Initialize sections of configuration
        self._init_paths()
        self._init_features()
        self._init_graph()
        self._init_training()
        self._init_device()
        self._init_experiment()

        # Override defaults with provided dictionary
        if cfg_dict:
            self.__dict__.update(cfg_dict)

    # -----------------------------
    # Initialization helper methods
    # -----------------------------
    def _init_paths(self):
        """Initialize file paths and create directories if needed."""
        self.data_path = "data/"
        os.makedirs(self.data_path, exist_ok=True)

        self.raw_data_file_path = os.path.join(self.data_path, "raw_kc.csv")
        self.processed_data_file_path = os.path.join(self.data_path, "processed_data.csv")
        self.spatial_features_file_path = os.path.join(self.data_path, "spatial_features.csv")
        self.feature_statistics_file_path = os.path.join(self.data_path, "feature_statistics.csv")
        self.train_data_file_path = os.path.join(self.data_path, "train_data.csv")
        self.val_data_file_path = os.path.join(self.data_path, "val_data.csv")
        self.test_data_file_path = os.path.join(self.data_path, "test_data.csv")

    def _init_features(self):
        """Initialize features used for training and target label."""
        self.geo_features = ["lat", "long"]
        self.embedding_features = [
            "bedrooms", "bathrooms", "sqft_living", "floors",
            "grade", "condition", "sqft_above", "sqft_basement"
        ]
        self.engineered_features = [
            "total_sqft", "bath_bed_ratio", "house_age", "years_since_reno",
            "has_basement", "has_been_renovated", "is_luxury", "space_efficiency",
            "neighborhood_price_mean", "neighborhood_price_median", "neighborhood_price_std"
        ]
        self.label_col = "price"

    def _init_graph(self):
        """Initialize graph construction parameters."""
        self.graph_type = "radius"
        self.radius1_k = 30
        self.radius2_k = 20
        self.radius3_k = 0
        self.top_k_sim = 7

    def _init_training(self):
        """Initialize training hyperparameters."""
        self.batch_size = 32
        self.learning_rate = 0.001
        self.max_epochs = 50
        self.criterion = "MSELoss"
        self.optimizer = "Adam"

    def _init_device(self):
        """Set computation device (CUDA if available)."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_device = self.device

    def _init_experiment(self):
        """Initialize experiment settings and output directory."""
        self.models_to_run = ["SimpleGCN", "SimpleGAT", "MultiLayerGCN"]
        self.save_path = "experiments/"
        os.makedirs(self.save_path, exist_ok=True)

    # -----------------------------
    # Static loader
    # -----------------------------
    @staticmethod
    def from_file(path: str):
        """
        Load configuration from a YAML or JSON file.

        Args:
            path (str): Path to configuration file.

        Returns:
            Config: Config object initialized with file contents.

        Raises:
            ValueError: If file is not YAML or JSON.
        """
        if path.endswith((".yaml", ".yml")):
            with open(path, "r") as f:
                return Config(yaml.safe_load(f))
        elif path.endswith(".json"):
            with open(path, "r") as f:
                return Config(json.load(f))
        else:
            raise ValueError("Config file must be YAML or JSON")
