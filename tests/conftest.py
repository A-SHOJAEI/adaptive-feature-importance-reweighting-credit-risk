"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="TARGET")

    return X_df, y_series


@pytest.fixture
def sample_train_val_data():
    """Generate sample train and validation data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )

    # Split into train and validation
    split_idx = 800
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # Convert to DataFrame/Series
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    y_train_series = pd.Series(y_train, name="TARGET")
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    y_val_series = pd.Series(y_val, name="TARGET")

    return X_train_df, y_train_series, X_val_df, y_val_series


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "data": {
            "dataset_name": "home_credit",
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "sample_frac": 0.1,
            "min_samples": 1000,
        },
        "model": {
            "type": "lightgbm",
            "lightgbm": {
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 10,
                "random_state": 42,
            },
        },
        "reweighting": {
            "enabled": False,
        },
        "curriculum": {
            "enabled": False,
        },
        "training": {
            "num_boost_round": 5,
            "early_stopping_rounds": 3,
            "verbose_eval": 1,
        },
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
