# Functions for data cleaning (missing values, outliers, scaling, etc.)

import numpy as np
import pandas as pd


def load_data(file_path):
    # Load data from CSV file
    return pd.read_csv(file_path)


def scale_features(X):
    # Scale features
    mean = np.mean(X, axis=0)
    std = np.std(X,axis=0)
    X_scaled = (X - mean) / std
    return X_scaled


def train_test_split(X, y, test_size=0.2, random_state=None):
    # Split the data into training and testing sets
    if random_state:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
