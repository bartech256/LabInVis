import torch
import os

class Config:
    def __init__(self):
        # General parameters
        self.default_random_seed = 42

        # Data loading parameters
        self.data_path = "data/"
        self.raw_data_file_path = os.path.join(self.data_path, "raw_data.csv")
        self.processed_data_file_path = os.path.join(self.data_path, "processed_data.csv")
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
        self.raw_embedding_features = [
            "bedrooms", "bathrooms", "sqft_living", "floors",
            "grade", "condition", "sqft_above", "sqft_basement"
        ]
        self.engineered_embedding_features = [
            "bedrooms", "bathrooms", "sqft_living", "floors",
            "grade", "condition", "sqft_above", "sqft_basement",
            # New engineered features
            "total_sqft", "bath_bed_ratio", "house_age", "years_since_reno",
            "has_basement", "has_been_renovated", "is_luxury", "space_efficiency",
            "neighborhood_price_mean", "neighborhood_price_median", "neighborhood_price_std"
        ]
        self.label_col = "price"

        # Experiment parameters
        self.experiment_name = "default_experiment"
        self.save_experiment_path = "experiments/"
        self.GNN_model_type = "SimpleGNN"
        self.GNN_model_params = {}
        self.regression_model_type = "LinearRegression"
        self.regression_model_params = {}
        self.optimizer = "Adam"
        self.criterion = "MSELoss"
        self.max_epochs = 50  # Reduced from 100 for testing
        self.train_loss_stop_threshold = 0.01
        self.batch_size = 32
        self.learning_rate = 0.001

        self.default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.what_to_visualize = ["training_loss", "validation_metrics", "graph_structure"] # Options can be seen in visualizer.py