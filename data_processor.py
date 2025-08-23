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
        self.cfg = cfg
        self.feature_scaler = None
        self.label_scaler = None

    def load_raw(self):
        return pd.read_csv(self.cfg.raw_data_file_path)

    def engineer_features(self, df):
        df = df.copy()
        df["total_sqft"] = df["sqft_above"] + df["sqft_basement"]
        df["bath_bed_ratio"] = df["bathrooms"] / np.maximum(df["bedrooms"], 1)
        df["house_age"] = 2015 - df["yr_built"]
        df["years_since_reno"] = np.where(df["yr_renovated"] == 0,
                                          df["house_age"],
                                          2015 - df["yr_renovated"])
        df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
        df["has_been_renovated"] = (df["yr_renovated"] > 0).astype(int)
        df["is_luxury"] = ((df["grade"] >= 10) | (df["waterfront"] == 1) | (df["view"] >= 3)).astype(int)
        df["space_efficiency"] = df["sqft_living"] / np.maximum(df["sqft_lot"], 1)
        return df

    def neighborhood_features(self, df):
        lat_bins = pd.cut(df['lat'], bins=50, labels=False)
        long_bins = pd.cut(df['long'], bins=50, labels=False)
        df["neighborhood_id"] = lat_bins * 50 + long_bins
        stats = df.groupby("neighborhood_id")["price"].agg(["mean","median","std"]).reset_index()
        stats.columns = ["neighborhood_id","neighborhood_price_mean","neighborhood_price_median","neighborhood_price_std"]
        df = df.merge(stats, on="neighborhood_id", how="left")
        return df

    def preprocess(self, df):
        features = self.cfg.embedding_features + self.cfg.engineered_features
        X = df[features].values
        y = df[self.cfg.label_col].values.reshape(-1, 1)
        self.feature_scaler = MinMaxScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        self.label_scaler = MinMaxScaler()
        y_scaled = self.label_scaler.fit_transform(y).flatten()
        return X_scaled, y_scaled

    def split(self, X, y, ratios=(0.7,0.15,0.15)):
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        train_end = int(ratios[0]*n)
        val_end = train_end + int(ratios[1]*n)
        return (X[indices[:train_end]], y[indices[:train_end]]), \
               (X[indices[train_end:val_end]], y[indices[train_end:val_end]]), \
               (X[indices[val_end:]], y[indices[val_end:]])

    def load_or_create_data(self):
        if os.path.exists(self.cfg.processed_data_file_path):
            df = pd.read_csv(self.cfg.processed_data_file_path)
        else:
            df = self.load_raw()
            df = self.engineer_features(df)
            df = self.neighborhood_features(df)
            df.to_csv(self.cfg.processed_data_file_path, index=False)
        X, y = self.preprocess(df)
        return self.split(X, y)
