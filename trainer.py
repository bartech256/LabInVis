from config import Config


class Trainer:
    """Trainer class for training and validating a model."""
    def __init__(self, model, optimizer, criterion, cfg: Config):
        # attach SummaryWriter, device, early stopping, etc.

    def train_epoch(self, loader):
        """One epoch of training."""

    def validate(self, loader):
        """
        Compute val loss, metrics.
        use the Evaluator class to compute metrics.
        """

    def fit(self, train_loader, val_loader):
        """Loop over epochs, call train_epoch + validate, save best ckpt."""

    def save_model(self):
        """Save the model state to a file."""

    def load_model(self):
        """Load the model state from a file."""

    def adjust_learning_rate(self, epoch, cfg: Config):
        """Adjust the learning rate based on the epoch."""