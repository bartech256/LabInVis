import torch

class Config:
    """
    Configuration class for the whole pipeline.
    It contains parameters for data loading, graph construction,
    model training, and logging.
    It is designed to be easily extensible for different datasets,
    models, and training configurations.
    """
    def __init__(self):
        # General parameters
        self.default_random_seed = 42

        # Data loading parameters
        self.data_path = "data/"
        self.raw_data_file_path = self.data_path + "raw_data.csv"
        self.processed_data_file_path = self.data_path + "processed_data.csv"
        self.spatial_features_file_path = self.data_path + "spatial_features.csv"
        self.feature_statistics_file_path = self.data_path + "feature_statistics.csv"
        self.train_data_file_path = self.data_path + "train_data.csv"
        self.val_data_file_path = self.data_path + "val_data.csv"
        self.test_data_file_path = self.data_path + "test_data.csv"
        self.default_train_val_test_split = (0.6, 0.2, 0.2)

        # Graph construction parameters
        self.graph_type = "spatial"

        # experiment parameters
        self.experiment_name = "default_experiment"
        self.save_experiment_path = "experiments/"
        self.default_model = "SimpleGNN"
        self.default_optimizer = "Adam"
        self.default_criterion = "MSELoss"
        self.default_epochs = 100
        self.default_batch_size = 32
        self.default_learning_rate = 0.001
        self.default_device = "cuda" if torch.cuda.is_available() else "cpu"
