"""
Responsibility:
- Build graph from processed data (X, y, geo).
- Supports distance-based edge construction:
  - Group 1: 0-5 miles (connect top-k similar by full features)
  - Group 2: 5-20 miles (connect top-k similar by structural features)
  - Group 3: 20+ miles (connect top-k similar by quality features)
"""

import numpy as np
from torch_geometric.data import Data
import torch
from sklearn.metrics.pairwise import cosine_similarity, haversine_distances


class GraphBuilder:
    """
    GraphBuilder constructs a graph using 3 distance-based groups:
    - Group 1 (0-5 miles): Connect to top-k most similar nodes by embedding
    - Group 2 (5-20 miles): Connect to top-k most similar nodes by structural similarity
    - Group 3 (20+ miles): Connect to top-k most similar nodes by quality similarity
    """

    def __init__(self, X: np.ndarray, geo: np.ndarray, y: np.ndarray, cfg):
        self.X = X
        self.geo = geo
        self.y = y
        self.cfg = cfg
        self.structural_features = self._extract_structural_features()
        self.quality_features = self._extract_quality_features()

        # Distance thresholds in miles from config
        self.radius1_miles = getattr(cfg, 'radius1', 5.0)
        self.radius2_miles = getattr(cfg, 'radius2', 20.0)

        print(f"GraphBuilder initialized with distance-based grouping")
        print(f"  Distance groups: [0-{self.radius1_miles} miles], [{self.radius1_miles}-{self.radius2_miles} miles], [{self.radius2_miles}+ miles]")
        print(f"  X shape: {X.shape}, geo shape: {geo.shape}, y shape: {y.shape}")
        print(f"  Structural features shape: {self.structural_features.shape}")
        print(f"  Quality features shape: {self.quality_features.shape}")

    def _extract_structural_features(self):
        structural_names = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_above']
        indices = [self.cfg.embedding_features.index(n) for n in structural_names if n in self.cfg.embedding_features]
        return self.X[:, indices] if indices else self.X[:, :self.X.shape[1] // 2]

    def _extract_quality_features(self):
        quality_names = ['grade', 'condition', 'sqft_basement']
        indices = [self.cfg.embedding_features.index(n) for n in quality_names if n in self.cfg.embedding_features]
        return self.X[:, indices] if indices else self.X[:, self.X.shape[1] // 2:]

    def _calculate_distance_matrix(self):
        """Calculate haversine distance matrix in miles"""
        # Convert to radians for haversine
        geo_radians = np.radians(self.geo)
        # haversine_distances returns distances in radians, multiply by Earth radius in miles
        earth_radius_miles = 3959.0
        distance_matrix = haversine_distances(geo_radians) * earth_radius_miles
        return distance_matrix

    def _get_distance_based_neighborhoods(self):
        """Group neighbors by distance thresholds from config"""
        distance_matrix = self._calculate_distance_matrix()
        n_nodes = self.geo.shape[0]

        neighborhoods = {}
        for i in range(n_nodes):
            distances = distance_matrix[i]

            # Create masks for each distance group (excluding self)
            mask_self = np.arange(n_nodes) != i
            mask_group1 = (distances <= self.radius1_miles) & mask_self  # 0 to radius1
            mask_group2 = (distances > self.radius1_miles) & (distances <= self.radius2_miles) & mask_self  # radius1 to radius2
            mask_group3 = (distances > self.radius2_miles) & mask_self  # radius2+

            neighborhoods[i] = {
                'group1': np.where(mask_group1)[0],  # 0 to radius1 miles
                'group2': np.where(mask_group2)[0],  # radius1 to radius2 miles
                'group3': np.where(mask_group3)[0]   # radius2+ miles
            }

        return neighborhoods

    def build(self) -> Data:
        neighborhoods = self._get_distance_based_neighborhoods()

        edge_index_g1 = self._build_group1_edges(neighborhoods)
        edge_index_g2 = self._build_group2_edges(neighborhoods)
        edge_index_g3 = self._build_group3_edges(neighborhoods)

        edge_index = torch.cat([edge_index_g1, edge_index_g2, edge_index_g3], dim=1)
        edge_index = self._remove_self_loops(self._remove_duplicate_edges(edge_index))

        print(f"Graph built with {edge_index.shape[1]} edges")
        print(f"  Group 1 (0-{self.radius1_miles} miles): {edge_index_g1.shape[1]} edges")
        print(f"  Group 2 ({self.radius1_miles}-{self.radius2_miles} miles): {edge_index_g2.shape[1]} edges")
        print(f"  Group 3 ({self.radius2_miles}+ miles): {edge_index_g3.shape[1]} edges")

        return Data(
            x=torch.tensor(self.X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(self.y, dtype=torch.float32).squeeze()
        )

    def _select_top_similar(self, base_vec, candidates, indices, top_k):
        """Select top-k most similar nodes from candidates"""
        if len(indices) == 0:
            return []

        candidate_features = candidates[indices]
        sims = cosine_similarity(base_vec.reshape(1, -1), candidate_features)[0]
        top_idx = sims.argsort()[::-1][:min(top_k, len(sims))]
        return indices[top_idx]

    def _build_group1_edges(self, neighborhoods):
        """Build edges for Group 1 (0 to radius1 miles) using full feature similarity"""
        if self.cfg.radius1_k == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_list = []
        top_k = getattr(self.cfg, 'top_k_sim', self.cfg.radius1_k)

        for i, neighbors in neighborhoods.items():
            candidates = neighbors['group1']
            if len(candidates) == 0:
                continue

            selected = self._select_top_similar(
                self.X[i], self.X, candidates, top_k
            )
            for j in selected:
                edge_list.append([i, j])

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    def _build_group2_edges(self, neighborhoods):
        """Build edges for Group 2 (radius1 to radius2 miles) using structural feature similarity"""
        if self.cfg.radius2_k == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_list = []
        top_k = getattr(self.cfg, 'top_k_sim', self.cfg.radius2_k)

        for i, neighbors in neighborhoods.items():
            candidates = neighbors['group2']
            if len(candidates) == 0:
                continue

            selected = self._select_top_similar(
                self.structural_features[i], self.structural_features, candidates, top_k
            )
            for j in selected:
                edge_list.append([i, j])

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    def _build_group3_edges(self, neighborhoods):
        """Build edges for Group 3 (radius2+ miles) using quality feature similarity"""
        if self.cfg.radius3_k == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_list = []
        top_k = getattr(self.cfg, 'top_k_sim', self.cfg.radius3_k)

        for i, neighbors in neighborhoods.items():
            candidates = neighbors['group3']
            if len(candidates) == 0:
                continue

            selected = self._select_top_similar(
                self.quality_features[i], self.quality_features, candidates, top_k
            )
            for j in selected:
                edge_list.append([i, j])

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    def _remove_self_loops(self, edge_index):
        """Remove self-loops from edge index"""
        return edge_index[:, edge_index[0] != edge_index[1]]

    def _remove_duplicate_edges(self, edge_index):
        """Remove duplicate edges from edge index"""
        edge_set = set()
        unique = []
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edge_set:
                edge_set.add(edge)
                unique.append(edge)
        return torch.tensor(unique, dtype=torch.long).t().contiguous() if unique else torch.empty((2, 0), dtype=torch.long)