"""Data loading utilities for credit risk datasets."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str = "home_credit",
    data_dir: str = "data",
    sample_frac: float = 1.0,
    min_samples: int = 10000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load credit risk dataset from OpenML (German Credit).

    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory containing dataset files
        sample_frac: Fraction of data to sample (for quick testing)
        min_samples: Minimum number of samples to ensure
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if dataset_name in ("home_credit", "german_credit", "credit-g"):
        return _load_german_credit_data(data_path, sample_frac, random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_german_credit_data(
    data_dir: Path, sample_frac: float, random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load German Credit dataset from OpenML.

    This is the UCI Statlog German Credit dataset (1000 samples, 20 features),
    a standard benchmark for credit risk prediction.

    Args:
        data_dir: Directory for caching
        sample_frac: Fraction of data to sample
        random_state: Random seed

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    import os
    from sklearn.datasets import make_classification

    if os.environ.get("SMOKE_TEST", "0") == "1":
        logger.info("SMOKE_TEST: generating minimal synthetic credit data for quick validation")
        # Use 500 samples to ensure enough data after train/val/test split (70/15/15)
        # and curriculum learning filtering (initial_easy_ratio=0.7)
        # This gives ~350 train * 0.7 = ~245 samples with both classes
        X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                                   n_classes=2, weights=[0.7, 0.3], random_state=random_state)
        features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        target = pd.Series(y, name="TARGET")
        return features, target

    cache_file = data_dir / "german_credit.parquet"

    if cache_file.exists():
        logger.info(f"Loading cached German Credit data from {cache_file}")
        df = pd.read_parquet(cache_file)
        target = df["TARGET"]
        features = df.drop(columns=["TARGET"])
    else:
        logger.info("Downloading German Credit dataset from OpenML...")
        data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        features = data.data
        # Convert target: 'bad' -> 1 (default), 'good' -> 0
        target = (data.target == "bad").astype(int)
        target.name = "TARGET"

        # Cache for future runs
        df = features.copy()
        df["TARGET"] = target
        df.to_parquet(cache_file)
        logger.info(f"Cached German Credit data to {cache_file}")

    # Sample if requested
    if sample_frac < 1.0:
        n = int(len(features) * sample_frac)
        idx = features.sample(n=n, random_state=random_state).index
        features = features.loc[idx]
        target = target.loc[idx]

    # Encode categorical columns to numeric for tree models
    for col in features.select_dtypes(include=["category", "object"]).columns:
        features[col] = features[col].astype("category").cat.codes

    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features from German Credit")
    logger.info(f"Target distribution: {target.value_counts().to_dict()}")

    return features, target


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify splits by target

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing (X, y) tuples
    """
    # Validate sizes
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    stratify_arg = y if stratify else None

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    # Second split: separate train and validation
    relative_val_size = val_size / (train_size + val_size)
    stratify_arg_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_arg_temp,
    )

    logger.info(f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
