"""Tests for data loading and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from adaptive_feature_importance_reweighting_credit_risk.data.loader import (
    load_dataset,
    split_data,
)
from adaptive_feature_importance_reweighting_credit_risk.data.preprocessing import (
    CreditDataPreprocessor,
)


class TestDataLoader:
    """Tests for data loading functions."""

    def test_load_dataset_synthetic(self):
        """Test loading synthetic dataset."""
        X, y = load_dataset(dataset_name="home_credit", min_samples=1000, random_state=42)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) >= 1000
        assert y.name == "TARGET"
        assert set(y.unique()).issubset({0, 1})

    def test_load_dataset_sample_frac(self):
        """Test loading with sample fraction."""
        X_full, y_full = load_dataset(
            dataset_name="home_credit", sample_frac=1.0, min_samples=1000, random_state=42
        )

        X_half, y_half = load_dataset(
            dataset_name="home_credit", sample_frac=0.5, min_samples=1000, random_state=42
        )

        assert len(X_half) < len(X_full)
        assert len(X_half) >= 1000

    def test_split_data(self, sample_data):
        """Test data splitting."""
        X, y = sample_data

        splits = split_data(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
        )

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Check sizes
        total_size = len(X_train) + len(X_val) + len(X_test)
        assert total_size == len(X)

        # Check proportions (with some tolerance)
        assert abs(len(X_train) / len(X) - 0.7) < 0.05
        assert abs(len(X_val) / len(X) - 0.15) < 0.05
        assert abs(len(X_test) / len(X) - 0.15) < 0.05

    def test_split_data_stratified(self, sample_data):
        """Test stratified splitting."""
        X, y = sample_data

        splits = split_data(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, stratify=True
        )

        _, y_train = splits["train"]
        _, y_val = splits["val"]
        _, y_test = splits["test"]

        # Check that class proportions are similar
        train_pos_ratio = y_train.mean()
        val_pos_ratio = y_val.mean()
        test_pos_ratio = y_test.mean()

        assert abs(train_pos_ratio - val_pos_ratio) < 0.1
        assert abs(train_pos_ratio - test_pos_ratio) < 0.1


class TestCreditDataPreprocessor:
    """Tests for credit data preprocessing."""

    def test_preprocessor_fit_transform(self, sample_data):
        """Test preprocessor fit and transform."""
        X, y = sample_data

        preprocessor = CreditDataPreprocessor(
            scale_features=True,
            handle_missing="median",
            encode_categoricals=True,
            create_interactions=False,
        )

        X_transformed = preprocessor.fit_transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)
        assert preprocessor.feature_names_ is not None

    def test_preprocessor_transform_consistency(self, sample_data):
        """Test that transform is consistent."""
        X, y = sample_data

        preprocessor = CreditDataPreprocessor()
        X_train = X[:800]
        X_test = X[800:]

        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Check that column names match
        assert list(X_train_transformed.columns) == list(X_test_transformed.columns)

    def test_preprocessor_handle_missing(self, sample_data):
        """Test missing value handling."""
        X, y = sample_data

        # Introduce missing values
        X_missing = X.copy()
        X_missing.iloc[0:10, 0] = np.nan

        preprocessor = CreditDataPreprocessor(handle_missing="median")
        X_transformed = preprocessor.fit_transform(X_missing)

        # Check no missing values remain
        assert not X_transformed.isnull().any().any()

    def test_preprocessor_create_interactions(self):
        """Test interaction feature creation."""
        # Create data with specific feature names
        X = pd.DataFrame(
            {
                "INCOME": np.random.uniform(1000, 5000, 100),
                "CREDIT_AMOUNT": np.random.uniform(500, 2000, 100),
                "AGE": np.random.uniform(20, 70, 100),
            }
        )

        preprocessor = CreditDataPreprocessor(create_interactions=True)
        X_transformed = preprocessor.fit_transform(X)

        # Check that interaction features were created
        assert "INCOME_CREDIT_RATIO" in X_transformed.columns
        assert len(X_transformed.columns) > len(X.columns)

    def test_preprocessor_categorical_encoding(self):
        """Test categorical feature encoding."""
        X = pd.DataFrame(
            {
                "numeric_feature": np.random.randn(100),
                "categorical_feature": np.random.choice(["A", "B", "C"], 100),
            }
        )

        preprocessor = CreditDataPreprocessor(encode_categoricals=True)
        X_transformed = preprocessor.fit_transform(X)

        # Check that categorical feature was encoded
        assert X_transformed["categorical_feature"].dtype in [np.int64, np.int32]

    def test_preprocessor_get_feature_names(self, sample_data):
        """Test getting feature names."""
        X, y = sample_data

        preprocessor = CreditDataPreprocessor()
        preprocessor.fit_transform(X)

        feature_names = preprocessor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
