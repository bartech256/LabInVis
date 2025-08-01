import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from torch_geometric.data import Data
from graph_builder import GraphBuilder
import torch
import time

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

    def engineer_features(self, df):
        """Add engineered features that might improve prediction"""
        df = df.copy()

        print("Engineering new features")

        # Price per sqft
        df['price_per_sqft'] = df['price'] / df['sqft_living']

        # Total sqft
        df['total_sqft'] = df['sqft_above'] + df['sqft_basement']

        # Bathroom to bedroom ratio
        df['bath_bed_ratio'] = df['bathrooms'] / np.maximum(df['bedrooms'], 1)

        # Age of house assuming current year is 2015 based on data
        current_year = 2015
        df['house_age'] = current_year - df['yr_built']

        # Years since renovation 0 if never renovated
        df['years_since_reno'] = np.where(df['yr_renovated'] == 0,
                                          df['house_age'],
                                          current_year - df['yr_renovated'])

        # Binary features
        df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
        df['has_been_renovated'] = (df['yr_renovated'] > 0).astype(int)
        df['is_luxury'] = ((df['grade'] >= 10) | (df['waterfront'] == 1) | (df['view'] >= 3)).astype(int)

        # Living space efficiency living sqft / lot sqft
        df['space_efficiency'] = df['sqft_living'] / np.maximum(df['sqft_lot'], 1)

        print(f"Added 9 engineered features")
        return df

    def create_neighborhood_features(self, df):
        """Create neighborhood-based features using geographic proximity"""
        print("Creating neighborhood features...")

        # Create a simple grid-based neighborhood (0.01 degree bins)
        lat_bins = pd.cut(df['lat'], bins=50, labels=False)
        long_bins = pd.cut(df['long'], bins=50, labels=False)
        df['neighborhood_id'] = lat_bins * 50 + long_bins

        # Calculate neighborhood statistics
        neighborhood_stats = df.groupby('neighborhood_id')['price'].agg(
            ['mean', 'median', 'std', 'count']).reset_index()
        neighborhood_stats.columns = ['neighborhood_id', 'neighborhood_price_mean', 'neighborhood_price_median',
                                      'neighborhood_price_std', 'neighborhood_count']

        # Merge back with original data
        df = df.merge(neighborhood_stats, on='neighborhood_id', how='left')

        # Fill NaN values
        df['neighborhood_price_std'] = df['neighborhood_price_std'].fillna(df['price'].std())

        print(f"Created neighborhood features for {df['neighborhood_id'].nunique()} neighborhoods")
        return df

    def create_processed_data(self, force_recreate=False) -> Data:
        """
        Load, preprocess, and return the processed data as a Data object.
        if the processed data file exists, it will load from there.
        Otherwise, it will load the raw data, preprocess it, and save the processed data.
        :return:
        """

        if os.path.exists(self.cfg.processed_data_file_path) and not force_recreate:
            print(f"Data already created at: {self.cfg.processed_data_file_path}")

        else:
            print(f"Creating processed data from raw data at: {self.cfg.raw_data_file_path}")
            # Load raw data
            start_time = time.time()
            df = self.load_raw()
            load_time = time.time() - start_time

            print(f"Data loaded in {load_time:.2f}s")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {len(df.columns)}")

            # Show basic statistics
            print(f"\nBasic Statistics:")
            print(f"   Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
            print(f"   Mean price: ${df['price'].mean():,.0f}")
            print(f"   Median price: ${df['price'].median():,.0f}")

            # Feature engineering
            feature_start = time.time()
            print(f"\nApplying Feature Engineering...")
            df = self.engineer_features(df)
            df = self.create_neighborhood_features(df)
            feature_time = time.time() - feature_start

            print(f"Feature engineering completed in {feature_time:.2f}s")
            print(f"   Enhanced shape: {df.shape}")

            # Show sample of new features
            new_features = ['price_per_sqft', 'house_age', 'bath_bed_ratio', 'is_luxury', 'neighborhood_price_mean']
            print(f"\nSample of new features:")
            print(df[new_features].head())

            print(f"\nData loading timing:")
            print(f"   Raw loading: {load_time:.2f}s")
            print(f"   Feature engineering: {feature_time:.2f}s")
            print(f"   Total: {load_time + feature_time:.2f}s")

            spatial_features_start = time.time()
            print(f"\nCreating spatial features...")
            self.create_spatial_features(df)

            X, y = self.preprocess(df)
            print(f" Preprocessing completed in {time.time() - spatial_features_start:.2f}s")
            print(f"   Features shape: {X.shape}")
            print(f"   Labels shape: {y.shape}")

            # Save processed data


            print("creating train/val/test splits...")
            train_ratio, val_ratio, test_ratio = self.cfg.default_train_val_test_split

            # Create indices and split
            n_nodes = len(X)
            indices = np.arange(n_nodes)
            np.random.shuffle(indices)

            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
            train_size = int(train_ratio * n_nodes)
            val_size = int(val_ratio * n_nodes)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Build boolean masks
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True

            print(f"Split created:")
            print(f"   Train: {train_mask.sum():,} samples ({train_mask.sum() / n_nodes * 100:.1f}%)")
            print(f"   Val:   {val_mask.sum():,} samples ({val_mask.sum() / n_nodes * 100:.1f}%)")
            print(f"   Test:  {test_mask.sum():,} samples ({test_mask.sum() / n_nodes * 100:.1f}%)")

            # save a csv file with for each train, val and test

