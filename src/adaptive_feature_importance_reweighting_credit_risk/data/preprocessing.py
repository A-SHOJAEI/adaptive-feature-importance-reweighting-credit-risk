"""Data preprocessing for credit risk prediction."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class CreditDataPreprocessor:
    """
    Preprocessor for credit risk data with feature engineering and encoding.

    Handles missing values, categorical encoding, feature scaling, and
    feature engineering specific to credit risk prediction.
    """

    def __init__(
        self,
        scale_features: bool = True,
        handle_missing: str = "median",
        encode_categoricals: bool = True,
        create_interactions: bool = True,
        remove_outliers: bool = False,
        outlier_std: float = 5.0,
    ):
        """
        Initialize preprocessor.

        Args:
            scale_features: Whether to scale numerical features
            handle_missing: Method for handling missing values ('median', 'mean', 'drop')
            encode_categoricals: Whether to encode categorical variables
            create_interactions: Whether to create interaction features
            remove_outliers: Whether to remove outliers
            outlier_std: Number of standard deviations for outlier detection
        """
        self.scale_features = scale_features
        self.handle_missing = handle_missing
        self.encode_categoricals = encode_categoricals
        self.create_interactions = create_interactions
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: dict = {}
        self.feature_names_: Optional[List[str]] = None
        self.categorical_features_: Optional[List[str]] = None
        self.numerical_features_: Optional[List[str]] = None
        self.fill_values_: dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditDataPreprocessor":
        """
        Fit preprocessor on training data.

        Args:
            X: Features DataFrame
            y: Target Series (unused, for compatibility)

        Returns:
            self
        """
        X = X.copy()

        # Identify feature types
        self.categorical_features_ = self._identify_categorical_features(X)
        self.numerical_features_ = [
            col for col in X.columns if col not in self.categorical_features_
        ]

        logger.info(
            f"Identified {len(self.numerical_features_)} numerical "
            f"and {len(self.categorical_features_)} categorical features"
        )

        # Compute fill values for missing data
        if self.handle_missing in ["median", "mean"]:
            for col in self.numerical_features_:
                if self.handle_missing == "median":
                    self.fill_values_[col] = X[col].median()
                else:
                    self.fill_values_[col] = X[col].mean()

            for col in self.categorical_features_:
                self.fill_values_[col] = X[col].mode()[0] if len(X[col].mode()) > 0 else "missing"

        # Fit label encoders for categorical features
        if self.encode_categoricals:
            for col in self.categorical_features_:
                le = LabelEncoder()
                # Handle missing values before encoding
                non_null_values = X[col].dropna()
                if len(non_null_values) > 0:
                    le.fit(non_null_values.astype(str))
                    self.label_encoders[col] = le

        # Fit scaler for numerical features
        if self.scale_features:
            X_numeric = X[self.numerical_features_].fillna(0)
            self.scaler = StandardScaler()
            self.scaler.fit(X_numeric)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Features DataFrame

        Returns:
            Transformed DataFrame
        """
        X = X.copy()

        # Handle missing values
        if self.handle_missing == "drop":
            X = X.dropna()
        elif self.handle_missing in ["median", "mean"]:
            for col, fill_value in self.fill_values_.items():
                if col in X.columns:
                    X[col] = X[col].fillna(fill_value)

        # Encode categorical features
        if self.encode_categoricals:
            for col in self.categorical_features_:
                if col in X.columns and col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Convert to string and handle unseen categories
                    X[col] = X[col].fillna("missing").astype(str)
                    X[col] = X[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # Remove outliers if requested (only for training)
        if self.remove_outliers:
            X = self._remove_outliers(X)

        # Create interaction features
        if self.create_interactions:
            X = self._create_interaction_features(X)

        # Scale numerical features
        if self.scale_features and self.scaler is not None:
            numeric_cols = [col for col in self.numerical_features_ if col in X.columns]
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        # Handle inf and very large values
        X = self._clip_extreme_values(X)

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.

        Args:
            X: Features DataFrame
            y: Target Series (unused, for compatibility)

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)

    def _identify_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        Identify categorical features in the DataFrame.

        Args:
            X: Features DataFrame

        Returns:
            List of categorical feature names
        """
        categorical = []

        for col in X.columns:
            # Check if dtype is object or category
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                categorical.append(col)
            # Check if integer column has few unique values (likely categorical)
            elif X[col].dtype in ["int64", "int32"] and X[col].nunique() < 20:
                categorical.append(col)

        return categorical

    def _remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers based on z-score for numerical features.

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with outliers removed
        """
        X = X.copy()

        for col in self.numerical_features_:
            if col in X.columns:
                z_scores = np.abs((X[col] - X[col].mean()) / (X[col].std() + 1e-8))
                X = X[z_scores < self.outlier_std]

        logger.info(f"Removed outliers, {len(X)} samples remaining")
        return X

    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for credit risk prediction.

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with additional interaction features
        """
        X = X.copy()

        # Income-related interactions (use safe division)
        if "INCOME" in X.columns and "CREDIT_AMOUNT" in X.columns:
            X["INCOME_CREDIT_RATIO"] = X["INCOME"] / (X["CREDIT_AMOUNT"] + 1e-8)

        if "INCOME" in X.columns and "CREDIT_ANNUITY" in X.columns:
            X["INCOME_ANNUITY_RATIO"] = X["INCOME"] / (X["CREDIT_ANNUITY"] + 1e-8)

        # Age-related interactions
        if "AGE" in X.columns and "EMPLOYED_YEARS" in X.columns:
            X["EMPLOYMENT_AGE_RATIO"] = X["EMPLOYED_YEARS"] / (X["AGE"] + 1e-8)

        # Credit utilization
        if "CREDIT_UTILIZATION" in X.columns and "CREDIT_AMOUNT" in X.columns:
            X["CREDIT_BURDEN"] = X["CREDIT_UTILIZATION"] * X["CREDIT_AMOUNT"]

        # External source combinations
        ext_sources = [col for col in X.columns if "EXT_SOURCE" in col]
        if len(ext_sources) >= 2:
            X["EXT_SOURCE_MEAN"] = X[ext_sources].mean(axis=1)
            X["EXT_SOURCE_STD"] = X[ext_sources].std(axis=1)
            X["EXT_SOURCE_MIN"] = X[ext_sources].min(axis=1)
            X["EXT_SOURCE_MAX"] = X[ext_sources].max(axis=1)

        # Previous credit history interactions
        if "PREV_LOANS_COUNT" in X.columns and "PREV_DEFAULTS" in X.columns:
            X["DEFAULT_RATE"] = X["PREV_DEFAULTS"] / (X["PREV_LOANS_COUNT"] + 1e-8)

        logger.info(f"Created interaction features, total features: {len(X.columns)}")
        return X

    def _clip_extreme_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clip extreme values (inf, -inf, and very large values) to prevent XGBoost errors.

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with clipped values
        """
        X = X.copy()

        # Replace inf/-inf with NaN, then fill with median/mean
        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill any new NaNs with 0 (conservative approach)
        X = X.fillna(0)

        # Clip very large values to a reasonable range
        # Use a threshold that XGBoost can handle (e.g., +/- 1e10)
        max_val = 1e10
        X = X.clip(lower=-max_val, upper=max_val)

        return X

    def get_feature_names(self) -> List[str]:
        """
        Get names of features after transformation.

        Returns:
            List of feature names
        """
        if self.feature_names_ is None:
            raise ValueError("Preprocessor has not been fitted yet")
        return self.feature_names_
