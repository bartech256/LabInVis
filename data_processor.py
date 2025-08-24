"""
Responsibility:
- Load raw KC dataset.
- Apply feature engineering and scaling.
- Create neighborhood features.
- Split data into train/val/test (only once).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class DataProcessor:
    def __init__(self, cfg):
        # Saves the config file inside (to know paths and properties)
        self.cfg = cfg
        self.feature_scaler = None
        self.label_scaler = None

    def load_raw(self):
        """Load raw dataset"""
        print(f"Loading raw data from: {self.cfg.raw_data_file_path}")
        df = pd.read_csv(self.cfg.raw_data_file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def engineer_features(self, df):
        """Add engineered features and handle edge cases"""
        df = df.copy()
        # Total area
        df["total_sqft"] = df["sqft_above"] + df["sqft_basement"]
        # Bathroom/bedroom ratio
        df["bath_bed_ratio"] = df["bathrooms"] / np.maximum(df["bedrooms"], 1)
        # Age of the house
        df["house_age"] = 2015 - df["yr_built"]
        # How many years have passed since it was renovated 
        #(or age of the house if it has not been renovated)
        df["years_since_reno"] = np.where(
            df["yr_renovated"] == 0,
            df["house_age"],
            2015 - df["yr_renovated"]
        )
        # Indicator for a basement
        df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
        # Indicator for renovation
        df["has_been_renovated"] = (df["yr_renovated"] > 0).astype(int)
        # Is the house considered luxurious
        df["is_luxury"] = (
            (df["grade"] >= 10) | (df["waterfront"] == 1) | (df["view"] >= 3)
        ).astype(int)
        # Space utilization
        df["space_efficiency"] = df["sqft_living"] / np.maximum(df["sqft_lot"], 1)
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    # Divide the map into 50×50 squares by lat/long
    # Calculate the mean, median, and standard deviation of prices in each square (neighborhood)
    # Add this to each house
    def neighborhood_features(self, df):
        """Create neighborhood-level aggregated features"""
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
        # Fill missing stats with global dataset statistics
        df["neighborhood_price_mean"] = df["neighborhood_price_mean"].fillna(df["price"].mean())
        df["neighborhood_price_median"] = df["neighborhood_price_median"].fillna(df["price"].median())
        df["neighborhood_price_std"] = df["neighborhood_price_std"].fillna(df["price"].std())
        return df

    def preprocess(self, df):
        """Scale features and labels, drop NaN rows if any"""
        features = self.cfg.embedding_features + self.cfg.engineered_features
        df = df.dropna(subset=features + [self.cfg.label_col]) 
        X = df[features].values
        y = df[self.cfg.label_col].values.reshape(-1, 1)
        # Feature scaling
        self.feature_scaler = MinMaxScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        # Label scaling
        self.label_scaler = MinMaxScaler()
        y_scaled = self.label_scaler.fit_transform(y).flatten()
        return X_scaled, y_scaled

    def train_val_test_split(self, X, y, ratios=(0.7, 0.15, 0.15)):
        """Split dataset into train/val/test"""
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        train_end = int(ratios[0] * n)
        val_end = train_end + int(ratios[1] * n)
        return (
            (X[indices[:train_end]], y[indices[:train_end]]),
            (X[indices[train_end:val_end]], y[indices[train_end:val_end]]),
            (X[indices[val_end:]], y[indices[val_end:]]),
        )
    # If the data already exists, it is loaded. Otherwise, it is recreated. 
    #This way, the data is created exactly once and is used for all inputs.
    def load_or_create_data(self):
        """Load processed data if exists, otherwise create it"""
        if os.path.exists(self.cfg.processed_data_file_path):
            print(f"Loading processed data from: {self.cfg.processed_data_file_path}")
            df = pd.read_csv(self.cfg.processed_data_file_path)
        else:
            print("Creating processed data from raw")
            df = self.load_raw()
            df = self.engineer_features(df)
            df = self.neighborhood_features(df)
            df.to_csv(self.cfg.processed_data_file_path, index=False)
            print(f"Saved processed data to: {self.cfg.processed_data_file_path}")

        X, y = self.preprocess(df)
        return self.train_val_test_split(X, y)
