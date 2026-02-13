"""Core adaptive ensemble model implementation."""

import logging
from typing import Dict, List, Optional

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class AdaptiveEnsembleModel(BaseEstimator, ClassifierMixin):
    """
    Ensemble model combining LightGBM, XGBoost, and CatBoost.

    Supports adaptive feature importance reweighting and curriculum learning.
    """

    def __init__(
        self,
        model_type: str = "ensemble",
        lightgbm_params: Optional[Dict] = None,
        xgboost_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        ensemble_weights: Optional[Dict] = None,
    ):
        """
        Initialize adaptive ensemble model.

        Args:
            model_type: Type of model ('ensemble', 'lightgbm', 'xgboost', 'catboost')
            lightgbm_params: Parameters for LightGBM
            xgboost_params: Parameters for XGBoost
            catboost_params: Parameters for CatBoost
            ensemble_weights: Weights for ensemble members
        """
        self.model_type = model_type
        self.lightgbm_params = lightgbm_params or {}
        self.xgboost_params = xgboost_params or {}
        self.catboost_params = catboost_params or {}
        self.ensemble_weights = ensemble_weights or {
            "lightgbm": 0.4,
            "xgboost": 0.35,
            "catboost": 0.25,
        }

        self.models_: Dict[str, object] = {}
        self.feature_names_: Optional[List[str]] = None
        self._feature_importances: Optional[np.ndarray] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: int = 0,
    ) -> "AdaptiveEnsembleModel":
        """
        Fit the ensemble model.

        Args:
            X: Features DataFrame
            y: Target Series
            sample_weight: Optional sample weights
            eval_set: Optional validation set as list of (X_val, y_val) tuples
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity level

        Returns:
            self
        """
        self.feature_names_ = X.columns.tolist()

        if self.model_type in ["ensemble", "lightgbm"]:
            self._fit_lightgbm(X, y, sample_weight, eval_set, early_stopping_rounds, verbose)

        if self.model_type in ["ensemble", "xgboost"]:
            self._fit_xgboost(X, y, sample_weight, eval_set, early_stopping_rounds, verbose)

        if self.model_type in ["ensemble", "catboost"]:
            self._fit_catboost(X, y, sample_weight, eval_set, early_stopping_rounds, verbose)

        # Compute ensemble feature importance
        self._compute_feature_importance()

        logger.info(f"Trained {self.model_type} model with {len(self.models_)} base learners")
        return self

    def _fit_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        eval_set: Optional[List],
        early_stopping_rounds: Optional[int],
        verbose: int,
    ) -> None:
        """Fit LightGBM model."""
        params = self.lightgbm_params.copy()

        # Create dataset
        train_data = lgb.Dataset(X, label=y, weight=sample_weight)

        eval_data = None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            eval_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        callbacks = []
        if verbose > 0:
            callbacks.append(lgb.log_evaluation(period=verbose))
        if early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        valid_sets = [eval_data] if eval_data is not None else None
        valid_names = ["valid"] if eval_data is not None else None

        model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.models_["lightgbm"] = model
        logger.debug("Fitted LightGBM model")

    def _fit_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        eval_set: Optional[List],
        early_stopping_rounds: Optional[int],
        verbose: int,
    ) -> None:
        """Fit XGBoost model."""
        params = self.xgboost_params.copy()

        # Extract training parameters
        n_estimators = params.pop("n_estimators", 100)
        n_jobs = params.pop("n_jobs", -1)
        random_state = params.pop("random_state", 42)

        # Create model
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            **params,
        )

        # Prepare fit parameters
        fit_params = {"sample_weight": sample_weight, "verbose": verbose > 0}

        if eval_set is not None and len(eval_set) > 0:
            fit_params["eval_set"] = eval_set

        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        # Train model
        model.fit(X, y, **fit_params)

        self.models_["xgboost"] = model
        logger.debug("Fitted XGBoost model")

    def _fit_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        eval_set: Optional[List],
        early_stopping_rounds: Optional[int],
        verbose: int,
    ) -> None:
        """Fit CatBoost model."""
        params = self.catboost_params.copy()

        # Extract parameters
        iterations = params.pop("iterations", 100)
        verbose_param = params.pop("verbose", 0)

        # Create model
        model = cb.CatBoostClassifier(iterations=iterations, verbose=verbose_param, **params)

        # Prepare fit parameters
        fit_params = {"sample_weight": sample_weight}

        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            fit_params["eval_set"] = (X_val, y_val)

        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        # Train model
        model.fit(X, y, **fit_params)

        self.models_["catboost"] = model
        logger.debug("Fitted CatBoost model")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        if not self.models_:
            raise ValueError("Model has not been fitted yet")

        predictions = []
        weights = []

        for model_name, model in self.models_.items():
            weight = self.ensemble_weights.get(model_name, 1.0)

            if model_name == "lightgbm":
                pred = model.predict(X)
            elif model_name == "xgboost":
                pred = model.predict_proba(X)[:, 1]
            elif model_name == "catboost":
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]

            predictions.append(pred)
            weights.append(weight)

        # Weighted average of predictions
        weights = np.array(weights)
        weights = weights / weights.sum()

        weighted_pred = np.average(predictions, axis=0, weights=weights)

        # Return in sklearn format [prob_class_0, prob_class_1]
        return np.vstack([1 - weighted_pred, weighted_pred]).T

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features DataFrame
            threshold: Classification threshold

        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def _compute_feature_importance(self) -> None:
        """Compute ensemble feature importance."""
        if not self.models_:
            return

        importances = []
        weights = []

        for model_name, model in self.models_.items():
            weight = self.ensemble_weights.get(model_name, 1.0)

            if model_name == "lightgbm":
                importance = model.feature_importance(importance_type="gain")
            elif hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            else:
                importance = np.ones(len(self.feature_names_)) / len(self.feature_names_)

            importances.append(importance)
            weights.append(weight)

        # Weighted average of importances
        weights = np.array(weights)
        weights = weights / weights.sum()

        self._feature_importances = np.average(importances, axis=0, weights=weights)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances."""
        if self._feature_importances is None:
            self._compute_feature_importance()
        return self._feature_importances

    @feature_importances_.setter
    def feature_importances_(self, value: np.ndarray) -> None:
        """Set feature importances."""
        self._feature_importances = value

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.

        Returns:
            DataFrame with feature names and importance values
        """
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise ValueError("Model has not been fitted yet")

        return pd.DataFrame(
            {"feature": self.feature_names_, "importance": self.feature_importances_}
        ).sort_values("importance", ascending=False)
