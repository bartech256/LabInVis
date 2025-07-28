import numpy as np
from typing import List
from torch_geometric.data import Data
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class GraphBuilder:
    """
    GraphBuilder constructs a graph using 3 concentric geographic radii:
    - Radius 1: Connect to top-k most similar nodes (by embedding) within closest geographic radius
    - Radius 2: Connect to top-k most similar nodes (by structural similarity)
    - Radius 3: Connect to top-k most similar nodes (by quality similarity)
    """

    def __init__(self, X: np.ndarray, geo: np.ndarray, y: np.ndarray, cfg):
        self.X = X
        self.geo = geo
        self.y = y
        self.cfg = cfg

        self.structural_features = self._extract_structural_features()
        self.quality_features = self._extract_quality_features()

        print(f"GraphBuilder initialized with concentric radii + similarity filtering")
        print(f"  X shape: {X.shape}, geo shape: {geo.shape}, y shape: {y.shape}")
        print(f"  Structural features shape: {self.structural_features.shape}")
        print(f"  Quality features shape: {self.quality_features.shape}")

    def _extract_structural_features(self):
        structural_names = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_above']
        indices = [self.cfg.embedding_features.index(n) for n in structural_names if n in self.cfg.embedding_features]
        return self.X[:, indices] if indices else self.X[:, :self.X.shape[1]//2]

    def _extract_quality_features(self):
        quality_names = ['grade', 'condition', 'sqft_basement']
        indices = [self.cfg.embedding_features.index(n) for n in quality_names if n in self.cfg.embedding_features]
        return self.X[:, indices] if indices else self.X[:, self.X.shape[1]//2:]

    def build(self) -> Data:
        neighborhoods = self._get_geographic_neighborhoods()
        edge_index_r1 = self._build_radius1_edges(neighborhoods)
        edge_index_r2 = self._build_radius2_edges(neighborhoods)
        edge_index_r3 = self._build_radius3_edges(neighborhoods)

        edge_index = torch.cat([edge_index_r1, edge_index_r2, edge_index_r3], dim=1)
        edge_index = self._remove_self_loops(self._remove_duplicate_edges(edge_index))

        return Data(
            x=torch.tensor(self.X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(self.y, dtype=torch.float32).squeeze()
        )

    def _get_geographic_neighborhoods(self):
        k1, k2, k3 = self.cfg.radius1_k, self.cfg.radius2_k, self.cfg.radius3_k
        total_neighbors = k1 + k2 + k3 + 1
        max_neighbors = min(total_neighbors, self.geo.shape[0])
        nbrs = NearestNeighbors(n_neighbors=max_neighbors, metric='haversine').fit(np.radians(self.geo))
        distances, indices = nbrs.kneighbors(np.radians(self.geo))

        neighborhoods = {}
        for i in range(self.geo.shape[0]):
            neighbors = indices[i][1:]
            neighborhoods[i] = {
                'radius1': neighbors[:k1],
                'radius2': neighbors[k1:k1+k2],
                'radius3': neighbors[k1+k2:k1+k2+k3]
            }
        return neighborhoods

    def _select_top_similar(self, base_vec, candidates, indices, top_k):
        if len(indices) == 0:
            return []
        sims = cosine_similarity(base_vec, candidates)[0]
        top_idx = sims.argsort()[::-1][:min(top_k, len(sims))]
        return indices[top_idx]

    def _build_radius1_edges(self, neighborhoods):
        edge_list = []
        top_k = getattr(self.cfg, 'top_k_sim', self.cfg.radius1_k)
        for i, neighbors in neighborhoods.items():
            candidates = neighbors['radius1']
            if len(candidates) == 0:
                continue
            selected = self._select_top_similar(self.X[i:i+1], self.X[candidates], candidates, top_k)
            for j in selected:
                edge_list.append([i, j])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    def _build_radius2_edges(self, neighborhoods):
        edge_list = []
        top_k = getattr(self.cfg, 'top_k_sim', self.cfg.radius2_k)
        for i, neighbors in neighborhoods.items():
            candidates = neighbors['radius2']
            if len(candidates) == 0:
                continue
            selected = self._select_top_similar(
                self.structural_features[i:i+1], self.structural_features[candidates], candidates, top_k
            )
            for j in selected:
                edge_list.append([i, j])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    def _build_radius3_edges(self, neighborhoods):
        edge_list = []
        top_k = getattr(self.cfg, 'top_k_sim', self.cfg.radius3_k)
        for i, neighbors in neighborhoods.items():
            candidates = neighbors['radius3']
            if len(candidates) == 0:
                continue
            selected = self._select_top_similar(
                self.quality_features[i:i+1], self.quality_features[candidates], candidates, top_k
            )
            for j in selected:
                edge_list.append([i, j])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    def _remove_self_loops(self, edge_index):
        return edge_index[:, edge_index[0] != edge_index[1]]

    def _remove_duplicate_edges(self, edge_index):
        edge_set = set()
        unique = []
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edge_set:
                edge_set.add(edge)
                unique.append(edge)
        return torch.tensor(unique, dtype=torch.long).t().contiguous() if unique else torch.empty((2, 0), dtype=torch.long)
