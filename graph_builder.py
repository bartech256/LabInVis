"""
Responsibility:
- Build a graph from processed features (X, y, geo)
- Supports distance-based edge construction:
  - Group 1: 0-5 miles, top-k by full features
  - Group 2: 5-20 miles, top-k by structural features
  - Group 3: 20+ miles, top-k by quality features
"""

import numpy as np
from torch_geometric.data import Data
import torch
from sklearn.metrics.pairwise import cosine_similarity, haversine_distances


class GraphBuilder:
    """Builds a graph with 3 distance-based groups using cosine similarity."""

    def __init__(self, X: np.ndarray, geo: np.ndarray, y: np.ndarray, cfg):
        self.X = X
        self.geo = geo
        self.y = y
        self.cfg = cfg

        # Extract subsets of features for structural and quality similarity
        self.structural_features = self._extract_structural_features()
        self.quality_features = self._extract_quality_features()

        # Distance thresholds in miles
        self.radius1_miles = getattr(cfg, 'radius1', 5.0)
        self.radius2_miles = getattr(cfg, 'radius2', 20.0)

        print(f"GraphBuilder initialized with distance-based grouping")
        print(f"Distance groups: [0-{self.radius1_miles}], [{self.radius1_miles}-{self.radius2_miles}], [{self.radius2_miles}+]")
        print(f"X shape: {X.shape}, geo shape: {geo.shape}, y shape: {y.shape}")
        print(f"Structural features shape: {self.structural_features.shape}")
        print(f"Quality features shape: {self.quality_features.shape}")

    def _extract_structural_features(self):
        """Select structural features for Group 2 similarity."""
        structural_names = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_above']
        indices = [self.cfg.embedding_features.index(n) for n in structural_names if n in self.cfg.embedding_features]
        return self.X[:, indices] if indices else self.X[:, :self.X.shape[1] // 2]

    def _extract_quality_features(self):
        """Select quality features for Group 3 similarity."""
        quality_names = ['grade', 'condition', 'sqft_basement']
        indices = [self.cfg.embedding_features.index(n) for n in quality_names if n in self.cfg.embedding_features]
        return self.X[:, indices] if indices else self.X[:, self.X.shape[1] // 2:]

    def _calculate_distance_matrix(self):
        """Compute Haversine distance matrix in miles."""
        geo_radians = np.radians(self.geo)
        earth_radius_miles = 3959.0
        return haversine_distances(geo_radians) * earth_radius_miles

    def _get_distance_based_neighborhoods(self):
        """Assign neighbors to distance groups."""
        distance_matrix = self._calculate_distance_matrix()
        n_nodes = self.geo.shape[0]
        neighborhoods = {}

        for i in range(n_nodes):
            distances = distance_matrix[i]
            mask_self = np.arange(n_nodes) != i

            neighborhoods[i] = {
                'group1': np.where((distances <= self.radius1_miles) & mask_self)[0],
                'group2': np.where((distances > self.radius1_miles) & (distances <= self.radius2_miles) & mask_self)[0],
                'group3': np.where((distances > self.radius2_miles) & mask_self)[0]
            }

        return neighborhoods

    def build(self) -> Data:
        """Build the final PyG Data graph."""
        neighborhoods = self._get_distance_based_neighborhoods()

        edge_index_g1 = self._build_edges_group(neighborhoods, 'group1', self.X, self.cfg.radius1_k)
        edge_index_g2 = self._build_edges_group(neighborhoods, 'group2', self.structural_features, self.cfg.radius2_k)
        edge_index_g3 = self._build_edges_group(neighborhoods, 'group3', self.quality_features, self.cfg.radius3_k)

        edge_index = torch.cat([edge_index_g1, edge_index_g2, edge_index_g3], dim=1)
        edge_index = self._remove_self_loops(self._remove_duplicate_edges(edge_index))

        print(f"Graph built with {edge_index.shape[1]} edges")
        return Data(
            x=torch.tensor(self.X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(self.y, dtype=torch.float32).squeeze()
        )

    def _build_edges_group(self, neighborhoods, group_name, feature_matrix, top_k):
        """Generic edge builder for a given distance group."""
        if top_k == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_list = []

        for i, neighbors in neighborhoods.items():
            candidates = neighbors[group_name]
            if len(candidates) == 0:
                continue

            sims = cosine_similarity(feature_matrix[i].reshape(1, -1), feature_matrix[candidates])[0]
            selected_idx = sims.argsort()[::-1][:min(top_k, len(sims))]
            for j in candidates[selected_idx]:
                edge_list.append([i, j])

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    @staticmethod
    def _remove_self_loops(edge_index):
        """Remove edges from a node to itself."""
        return edge_index[:, edge_index[0] != edge_index[1]]

    @staticmethod
    def _remove_duplicate_edges(edge_index):
        """Remove duplicate edges."""
        edge_set = set()
        unique_edges = []
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edge_set:
                edge_set.add(edge)
                unique_edges.append(edge)
        return torch.tensor(unique_edges, dtype=torch.long).t().contiguous() if unique_edges else torch.empty((2, 0), dtype=torch.long)
