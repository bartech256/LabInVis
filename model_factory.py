import models
import torch

class ModelFactory:
    """A factory class to create models based on configuration."""

    def create_model(self, config):
        """Create a model based on the provided configuration."""
        GNN_model_type = config.GNN_model_type
        GNN_model_params = config.GNN_model_params


        regression_model_type = config.regression_model_type
        regression_model_params = config.regression_model_params

        if GNN_model_type == 'SimpleGAT': # real from code
            GNN_model = models.SimpleGAT(**GNN_model_params)
        elif GNN_model_type == 'GraphSAGE': # example
            GNN_model = models.GraphSAGE(**GNN_model_params)
        elif GNN_model_type == 'GAT': # example
            GNN_model = models.GAT(**GNN_model_params)
        elif GNN_model_type == 'SimpleGCN': # example
            GNN_model = models.SimpleGCN(**GNN_model_params)
        elif GNN_model_type == None:
            GNN_model = None
        else:
            raise ValueError(f"Unknown GNN model type: {GNN_model_type}")

        if regression_model_type == 'LinearRegression':
            regression_model = models.LinearRegression(**regression_model_params)
        elif regression_model_type == 'AdvancedRegression': # example
            regression_model = models.AdvancedRegression(**regression_model_params)
        else:
            raise ValueError(f"Unknown regression model type: {regression_model_type}")

        if GNN_model is not None and regression_model is not None:
            # Combine GNN and regression model
            from models import CombinedModel
            return CombinedModel(GNN_model, regression_model)
        elif GNN_model is not None:
            # Only GNN model
            return GNN_model
        elif regression_model is not None:
            # Only regression model
            return regression_model
        else:
            raise ValueError("No model specified in the configuration.")

    def load_model(self, model_path):
        """Load a model from a given path."""
        model = torch.load(model_path)
        return model

    def save_model(self, model, model_path):
        """Save a model to a given path."""
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
