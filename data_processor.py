"""
Responsibility:
- Load raw KC dataset.
- Apply feature engineering and scaling.
- Create neighborhood features.
- Split data into train/val/test.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


class DataProcessor:
    """
    Handles loading, preprocessing, feature engineering, and splitting
    of the King County housing dataset.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (Config): Configuration object with paths, features, etc.
        """
        self.cfg = cfg
        self.feature_scaler = None
        self.label_scaler = None

    def load_raw(self):
        """
        Load the raw dataset from CSV.

        Returns:
            pd.DataFrame: Raw dataset.
        """
        print(f"Loading raw data from: {self.cfg.raw_data_file_path}")
        df = pd.read_csv(self.cfg.raw_data_file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def engineer_features(self, df):
        """
        Add engineered features to the dataset.

        Args:
            df (pd.DataFrame): Raw dataset.

        Returns:
            pd.DataFrame: Dataset with engineered features.
        """
        df = df.copy()
        # Basic area and ratio features
        df["total_sqft"] = df["sqft_above"] + df["sqft_basement"]
        df["bath_bed_ratio"] = df["bathrooms"] / np.maximum(df["bedrooms"], 1)
        df["house_age"] = 2015 - df["yr_built"]
        df["years_since_reno"] = np.where(
            df["yr_renovated"] == 0,
            df["house_age"],
            2015 - df["yr_renovated"]
        )
        df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
        df["has_been_renovated"] = (df["yr_renovated"] > 0).astype(int)
        df["is_luxury"] = (
            (df["grade"] >= 10) | (df["waterfront"] == 1) | (df["view"] >= 3)
        ).astype(int)
        df["space_efficiency"] = df["sqft_living"] / np.maximum(df["sqft_lot"], 1)

        # Handle infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def neighborhood_features(self, df):
        """
        Generate neighborhood-level aggregated features.

        Args:
            df (pd.DataFrame): Dataset with lat/long columns.

        Returns:
            pd.DataFrame: Dataset with neighborhood stats.
        """
        lat_bins = pd.cut(df["lat"], bins=50, labels=False)
        long_bins = pd.cut(df["long"], bins=50, labels=False)
        df["neighborhood_id"] = lat_bins * 50 + long_bins

        stats = df.groupby("neighborhood_id")["price"].agg(["mean", "median", "std"]).reset_index()
        stats.columns = [
            "neighborhood_id",
            "neighborhood_price_mean",
            "neighborhood_price_median",
            "neighborhood_price_std",
        ]
        df = df.merge(stats, on="neighborhood_id", how="left")

        df["neighborhood_price_mean"] = df["neighborhood_price_mean"].fillna(df["price"].mean())
        df["neighborhood_price_median"] = df["neighborhood_price_median"].fillna(df["price"].median())
        df["neighborhood_price_std"] = df["neighborhood_price_std"].fillna(df["price"].std())

        return df

    def preprocess(self, df):
        """
        Scale features and labels, drop rows with missing values.

        Args:
            df (pd.DataFrame): Dataset with all features.

        Returns:
            tuple: (X_scaled, y_scaled)
        """
        features = self.cfg.embedding_features + self.cfg.engineered_features
        df = df.dropna(subset=features + [self.cfg.label_col])

        X = df[features].values
        y = df[self.cfg.label_col].values.reshape(-1, 1)

        # Scale features
        self.feature_scaler = MinMaxScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        # Scale labels
        self.label_scaler = MinMaxScaler()
        y_scaled = self.label_scaler.fit_transform(y).flatten()

        return X_scaled, y_scaled

    def train_val_test_split(self, X, y, ratios=(0.7, 0.15, 0.15)):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            ratios (tuple): Split ratios (train, val, test).

        Returns:
            tuple: ((train_X, train_y), (val_X, val_y), (test_X, test_y))
        """
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)

        train_end = int(ratios[0] * n)
        val_end = train_end + int(ratios[1] * n)

        return (
            (X[indices[:train_end]], y[indices[:train_end]]),
            (X[indices[train_end:val_end]], y[indices[train_end:val_end]]),
            (X[indices[val_end:]], y[indices[val_end:]])
        )
