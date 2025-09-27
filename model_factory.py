"""
Responsibility:
- Factory class to create models based on configuration.
- Supports both GNNs (SimpleGCN, SimpleGAT, MultiLayerGCN) and baselines (MLPRegressor, LinearRegressionTorch).
- Can also combine GNN with regression via CombinedModel.
"""
import torch
import models

class ModelFactory:
    """A factory class to create models based on configuration."""

    def create_model(self, config) -> torch.nn.Module:
        """Create a model based on the provided configuration."""
        # Support both dict (YAML) and Config object
        if isinstance(config, dict):
            GNN_model_type = config.get("GNN_model_type", None)
            GNN_model_params = config.get("GNN_model_params", {})
            regression_model_type = config.get("regression_model_type", None)
            regression_model_params = config.get("regression_model_params", {})
            device = config.get("device", "cpu")
        else:
            GNN_model_type = getattr(config, "GNN_model_type", None)
            GNN_model_params = getattr(config, "GNN_model_params", {})
            regression_model_type = getattr(config, "regression_model_type", None)
            regression_model_params = getattr(config, "regression_model_params", {})
            device = getattr(config, "device", "cpu")

        # --- Create GNN model ---
        GNN_model = None
        if GNN_model_type == "SimpleGAT":
            GNN_model = models.SimpleGAT(**GNN_model_params)
        elif GNN_model_type == "SimpleGCN":
            GNN_model = models.SimpleGCN(**GNN_model_params)
        elif GNN_model_type == "MultiLayerGCN":
            GNN_model = models.MultiLayerGCN(**GNN_model_params)
        elif GNN_model_type == "GraphSAGE":
            GNN_model = models.GraphSAGE(**GNN_model_params)
        elif GNN_model_type is not None:
            raise ValueError(f"Unknown GNN model type: {GNN_model_type}")

        # --- Create regression/baseline model ---
        regression_model = None
        if regression_model_type == "MLPRegressor":
            regression_model = models.MLPRegressor(**regression_model_params)
        elif regression_model_type == "LinearRegressionTorch":
            regression_model = models.LinearRegressionTorch(**regression_model_params)
        elif regression_model_type is not None:
            raise ValueError(f"Unknown regression model type: {regression_model_type}")

        # --- Move models to device ---
        if GNN_model is not None:
            GNN_model.to(device)
        if regression_model is not None:
            regression_model.to(device)

        # --- Combine if both exist ---
        if GNN_model is not None and regression_model is not None:
            final_model = models.CombinedModel(GNN_model, regression_model)
        elif GNN_model is not None:
            final_model = GNN_model
        elif regression_model is not None:
            final_model = regression_model
        else:
            raise ValueError("No model specified in the configuration.")

        print(f"Created model: GNN={GNN_model_type}, Regression={regression_model_type} on device={device}")
        return final_model

    def load_model(self, model_path: str, map_location=None) -> torch.nn.Module:
        """Load a model from a given path."""
        model = torch.load(model_path, map_location=map_location)
        print(f"Model loaded from {model_path}")
        return model

    def save_model(self, model: torch.nn.Module, model_path: str) -> None:
        """Save a model to a given path."""
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
