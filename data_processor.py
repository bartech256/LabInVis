import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from torch_geometric.data import Data
from graph_builder import GraphBuilder
import torch

class DataProcessor:
    """
    A class to process data for GNN creation.
    Handles loading, preprocessing, splitting and saving.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_scaler = None
        self.label_scaler = None

    def load_raw(self) -> pd.DataFrame:
        print(f"Loading raw data from: {self.cfg.raw_data_file_path}")
        df = pd.read_csv(self.cfg.raw_data_file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print("Preprocessing data...")

        missing_features = [f for f in self.cfg.embedding_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")

        if self.cfg.label_col not in df.columns:
            raise ValueError(f"Label column '{self.cfg.label_col}' not found in dataset")

        original_len = len(df)
        df = df.dropna(subset=self.cfg.embedding_features + [self.cfg.label_col])
        print(f"Removed {original_len - len(df)} rows with NaN values")

        X = df[self.cfg.embedding_features].values
        y = df[self.cfg.label_col].values.reshape(-1, 1)

        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        self.feature_scaler = MinMaxScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        self.label_scaler = MinMaxScaler()
        y_scaled = self.label_scaler.fit_transform(y).flatten()

        print(f"Scaled features shape: {X_scaled.shape}")
        print(f"Scaled labels shape: {y_scaled.shape}")

        self.save_feature_statistics(df)

        return X_scaled, y_scaled

    def save_processed_data(self, X: np.ndarray, y: np.ndarray) -> None:
        df = pd.DataFrame(X, columns=self.cfg.embedding_features)
        df[self.cfg.label_col] = y
        df.to_csv(self.cfg.processed_data_file_path, index=False)
        print(f"Saved processed data to: {self.cfg.processed_data_file_path}")

    def load_processed_data(self) -> pd.DataFrame:
        return pd.read_csv(self.cfg.processed_data_file_path)

    def train_val_test_split(self, X, y) -> Tuple:
        n = len(X)
        train_end = int(n * self.cfg.default_train_val_test_split[0])
        val_end = train_end + int(n * self.cfg.default_train_val_test_split[1])

        print(f"Splitting data: train={train_end}, val={val_end-train_end}, test={n-val_end}")

        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:val_end], y[train_end:val_end]),
            (X[val_end:], y[val_end:])
        )

    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_geo = [f for f in self.cfg.geo_features if f not in df.columns]
        if missing_geo:
            raise ValueError(f"Missing geographic features: {missing_geo}")

        df_clean = df.dropna(subset=self.cfg.geo_features + self.cfg.embedding_features + [self.cfg.label_col])
        spatial_df = df_clean[self.cfg.geo_features].copy()
        spatial_df.to_csv(self.cfg.spatial_features_file_path, index=False)
        print(f"Saved spatial features to: {self.cfg.spatial_features_file_path}")
        print(f"Spatial features shape: {spatial_df.shape}")

        return spatial_df

    def save_feature_statistics(self, df: pd.DataFrame) -> None:
        stats = df[self.cfg.embedding_features].describe().transpose()
        stats.to_csv(self.cfg.feature_statistics_file_path)
        print(f"Saved feature statistics to: {self.cfg.feature_statistics_file_path}")

    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.cfg.embedding_features].describe().transpose()

    def create_data_object(self, X: np.ndarray, y: np.ndarray) -> Data:
        print("Creating graph data object...")

        try:
            geo_df = pd.read_csv(self.cfg.spatial_features_file_path)
            geo = geo_df[self.cfg.geo_features].values
        except FileNotFoundError:
            raise FileNotFoundError(f"Spatial features file not found: {self.cfg.spatial_features_file_path}. "
                                    "Make sure to call create_spatial_features() first.")

        if len(X) != len(geo) or len(X) != len(y):
            raise ValueError(f"Mismatched array sizes: X={len(X)}, geo={len(geo)}, y={len(y)}")

        builder = GraphBuilder(X, geo, y, self.cfg)
        data = builder.build()
        return data

    def create_split_graphs(self, X: np.ndarray, y: np.ndarray, train_mask: torch.Tensor,
                            val_mask: torch.Tensor, test_mask: torch.Tensor) -> Tuple[Data, Data, Data]:
        """
        Create separate graph objects for train, val, and test using corresponding node masks.
        Each graph includes only the nodes and edges within its subset.
        """
        print("🔄 Building separate graphs for Train / Val / Test")
        
        # Load geographic data
        try:
            geo_df = pd.read_csv(self.cfg.spatial_features_file_path)
            geo = geo_df[self.cfg.geo_features].values
        except FileNotFoundError:
            raise FileNotFoundError(f"Spatial features file not found: {self.cfg.spatial_features_file_path}. "
                                    "Make sure to call create_spatial_features() first.")

        def subset_data(mask: torch.Tensor):
            """Extract subset of data based on mask"""
            idx = mask.cpu().numpy().nonzero()[0]
            return X[idx], geo[idx], y[idx], idx

        def build_graph(X_sub, geo_sub, y_sub, subset_name):
            """Build graph for a data subset"""
            print(f"  Building {subset_name} graph with {len(X_sub)} nodes...")
            builder = GraphBuilder(X_sub, geo_sub, y_sub, self.cfg)
            return builder.build()

        # Extract subsets
        X_train, geo_train, y_train, train_idx = subset_data(train_mask)
        X_val, geo_val, y_val, val_idx = subset_data(val_mask)
        X_test, geo_test, y_test, test_idx = subset_data(test_mask)

        # Build separate graphs
        data_train = build_graph(X_train, geo_train, y_train, "train")
        data_val = build_graph(X_val, geo_val, y_val, "validation")
        data_test = build_graph(X_test, geo_test, y_test, "test")

        print(f"✅ Split graphs created:")
        print(f"   Train: {data_train.num_nodes} nodes, {data_train.num_edges} edges")
        print(f"   Val:   {data_val.num_nodes} nodes, {data_val.num_edges} edges")
        print(f"   Test:  {data_test.num_nodes} nodes, {data_test.num_edges} edges")

        return data_train, data_val, data_test