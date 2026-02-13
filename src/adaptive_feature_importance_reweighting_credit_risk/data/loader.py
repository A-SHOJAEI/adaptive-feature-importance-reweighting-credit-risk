"""Data loading utilities for credit risk datasets."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
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
    Load credit risk dataset.

    For demonstration, generates synthetic data mimicking Home Credit Default Risk
    dataset characteristics if real data is not available.

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

    if dataset_name == "home_credit":
        return _load_home_credit_data(data_path, sample_frac, min_samples, random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_home_credit_data(
    data_dir: Path, sample_frac: float, min_samples: int, random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Home Credit Default Risk dataset or generate synthetic data.

    Args:
        data_dir: Directory containing dataset files
        sample_frac: Fraction of data to sample
        min_samples: Minimum number of samples
        random_state: Random seed

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    train_file = data_dir / "application_train.csv"

    # Try to load real data if available
    if train_file.exists():
        logger.info(f"Loading Home Credit data from {train_file}")
        df = pd.read_csv(train_file)

        # Sample if requested
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)

        target = df["TARGET"]
        features = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

        logger.info(f"Loaded {len(df)} samples with {len(features.columns)} features")
        return features, target

    # Generate synthetic data if real data not available
    logger.warning("Real data not found. Generating synthetic Home Credit-like data.")
    return _generate_synthetic_credit_data(min_samples, random_state)


def _generate_synthetic_credit_data(
    n_samples: int = 50000, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic credit default data with realistic feature distributions.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    logger.info(f"Generating {n_samples} synthetic credit risk samples")

    np.random.seed(random_state)

    # Generate base features with classification task structure
    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_repeated=5,
        n_classes=2,
        weights=[0.92, 0.08],  # Imbalanced like real credit data
        flip_y=0.02,
        class_sep=0.6,
        random_state=random_state,
    )

    # Create realistic feature names and distributions
    feature_names = []
    feature_data = {}

    # Demographic features
    feature_data["AGE"] = np.clip(np.random.normal(43, 15, n_samples), 20, 70).astype(int)
    feature_data["GENDER"] = np.random.choice([0, 1], n_samples, p=[0.66, 0.34])
    feature_data["FAMILY_SIZE"] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    feature_data["CHILDREN"] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.45, 0.30, 0.20, 0.05])

    # Income and credit features (clip to avoid extreme values)
    feature_data["INCOME"] = np.clip(np.abs(np.random.lognormal(11.5, 0.8, n_samples)), 1e3, 1e7)
    feature_data["CREDIT_AMOUNT"] = np.clip(np.abs(np.random.lognormal(12.0, 0.7, n_samples)), 1e3, 1e8)
    feature_data["CREDIT_ANNUITY"] = feature_data["CREDIT_AMOUNT"] / np.random.uniform(12, 60, n_samples)
    feature_data["INCOME_CREDIT_RATIO"] = feature_data["INCOME"] / (feature_data["CREDIT_AMOUNT"] + 1e-8)

    # Employment features
    feature_data["EMPLOYED_DAYS"] = -np.abs(np.random.exponential(1000, n_samples))
    feature_data["EMPLOYED_YEARS"] = -feature_data["EMPLOYED_DAYS"] / 365
    feature_data["INCOME_TYPE"] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])

    # Housing and assets
    feature_data["OWNS_CAR"] = np.random.choice([0, 1], n_samples, p=[0.66, 0.34])
    feature_data["OWNS_REALTY"] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    feature_data["HOUSING_TYPE"] = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.8, 0.1])

    # Previous credit history
    feature_data["PREV_LOANS_COUNT"] = np.random.poisson(2, n_samples)
    feature_data["PREV_DEFAULTS"] = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.25, 0.05])
    feature_data["CREDIT_UTILIZATION"] = np.clip(np.random.beta(2, 5, n_samples), 0, 1)

    # External scores
    feature_data["EXT_SOURCE_1"] = np.clip(np.random.beta(2, 2, n_samples), 0, 1)
    feature_data["EXT_SOURCE_2"] = np.clip(np.random.beta(2, 2, n_samples), 0, 1)
    feature_data["EXT_SOURCE_3"] = np.clip(np.random.beta(2, 2, n_samples), 0, 1)

    # Regional features (clip to avoid extreme values)
    feature_data["REGION_POPULATION"] = np.clip(np.abs(np.random.lognormal(9, 1.5, n_samples)), 1e3, 1e7)
    feature_data["REGION_RATING"] = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.6, 0.2])

    # Add original synthetic features
    for i in range(X.shape[1]):
        feature_data[f"FEATURE_{i:02d}"] = X[:, i]

    # Create DataFrame
    df = pd.DataFrame(feature_data)
    target = pd.Series(y, name="TARGET")

    # Add some missing values to make it realistic
    for col in df.columns:
        if np.random.random() < 0.3:  # 30% of columns have missing values
            mask = np.random.random(n_samples) < 0.05  # 5% missing rate
            df.loc[mask, col] = np.nan

    logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")
    logger.info(f"Target distribution: {target.value_counts().to_dict()}")

    return df, target


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
