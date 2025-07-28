import torch
import os

class Config:
    def __init__(self):
        # General parameters
        self.default_random_seed = 42

        # Data loading parameters
        self.data_path = "data/"
        self.raw_data_file_path = "C:/Users/Yuval Rainis/Desktop/School/2025 B/Dlab/Project/kc_final.csv"
        self.processed_data_file_path = os.path.join(self.data_path, "processed_data.csv")
        self.spatial_features_file_path = os.path.join(self.data_path, "spatial_features.csv")
        self.feature_statistics_file_path = os.path.join(self.data_path, "feature_statistics.csv")
        self.train_data_file_path = os.path.join(self.data_path, "train_data.csv")
        self.val_data_file_path = os.path.join(self.data_path, "val_data.csv")
        self.test_data_file_path = os.path.join(self.data_path, "test_data.csv")
        self.default_train_val_test_split = (0.6, 0.2, 0.2)

        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)

        # Graph construction parameters
        self.graph_type = "spatial" 
        self.radius1_k = 30
        self.radius2_k = 20
        self.radius3_k = 0
        self.top_k_sim = 7
        self.geo_features = ["lat", "long"]
        self.embedding_features = [
            "bedrooms", "bathrooms", "sqft_living", "floors",
            "grade", "condition", "sqft_above", "sqft_basement"
        ]
        self.label_col = "price"

        # Experiment parameters
        self.experiment_name = "default_experiment"
        self.save_experiment_path = "experiments/"
        self.default_model = "SimpleGNN"
        self.default_optimizer = "Adam"
        self.default_criterion = "MSELoss"
        self.default_epochs = 50  # Reduced from 100 for testing
        self.default_batch_size = 32
        self.default_learning_rate = 0.001
        self.default_device = "cuda" if torch.cuda.is_available() else "cpu"