class DataProcessor:
    """
    A class to process data for GNN creation.
    It handles loading data from CSV files, preprocessing,
    and preparing it for graph construction.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_raw(self) -> pd.DataFrame:
        """Read CSV into DataFrame."""

    def load_processed_data(self) -> pd.DataFrame:
        """Load preprocessed data from CSV files."""

    def save_processed_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Save processed data to CSV files."""

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values, normalize, split features X and targets y."""

    def train_val_test_split(self, X, y) -> Tuple:
        """Return (X_train, y_train), (X_val, y_val), (X_test, y_test)."""

    def create_spatial_features (self df: pd.DataFrame) -> pd.DataFrame:
    """Create spatial features from the DataFrame."""

    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and return feature statistics like mean, std, min, max."""
