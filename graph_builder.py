class GraphBuilder:
    """Builds a graph from data using various methods."""
    def __init__(self, X: np.ndarray, cfg: Config):
        self.X = X
        self.cfg = cfg

    def build(self, method: str) -> torch_geometric.data.Data:
        """Dispatch to specific method."""
    def _build_knn(self, k: int) -> EdgeIndex:
        """k‑nearest neighbors graph."""
    def _build_distance_threshold(self, threshold: float) -> EdgeIndex:
        """Connect nodes within a distance."""
    def _build_correlation(self) -> EdgeIndex:
        """Connect based on feature correlation."""
    # … add other edge‑construction routines …