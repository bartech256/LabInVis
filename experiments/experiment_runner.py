class ExperimentRunner:
    """A class to run experiments with different configurations."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run_single_experiment(self, experiment_name: str):
        """Run a specific experiment based on the configuration."""

    def run_grid_search(self):
        """Run a grid search over multiple configurations."""

    def save_experiment_results(self):
        """
        Save the results of the experiments to a file.
        saves the model, metrics, and configuration.
        """

    def continue_experiment(self):
        """load an existing experiment and continue training."""

    def evaluate_experiment(self):
        """Evaluate the results of the experiment."""
        # This could involve loading the model and running it on a test set,
        # then calculating metrics like MAE, RMSE, etc. using the Evaluator class.

