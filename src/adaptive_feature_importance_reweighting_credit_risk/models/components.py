"""Custom components for adaptive feature importance reweighting."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


class AdaptiveFeatureReweighter:
    """
    Adaptive feature importance reweighting mechanism.

    Dynamically adjusts sample weights based on feature importance distributions
    computed via SHAP, permutation importance, or gradient-based attribution.
    """

    def __init__(
        self,
        strategy: str = "shap",
        top_k_features: int = 20,
        reweight_alpha: float = 0.3,
        temporal_decay: float = 0.95,
        segment_bins: int = 5,
        min_importance_threshold: float = 0.01,
    ):
        """
        Initialize adaptive reweighter.

        Args:
            strategy: Method for computing importance ('shap', 'permutation', 'gradient')
            top_k_features: Number of top features to focus on
            reweight_alpha: Strength of reweighting (0=none, 1=full)
            temporal_decay: Decay factor for historical importance
            segment_bins: Number of risk segments for stratified reweighting
            min_importance_threshold: Minimum importance to consider
        """
        self.strategy = strategy
        self.top_k_features = top_k_features
        self.reweight_alpha = reweight_alpha
        self.temporal_decay = temporal_decay
        self.segment_bins = segment_bins
        self.min_importance_threshold = min_importance_threshold

        self.feature_importance_history_: List[np.ndarray] = []
        self.current_importance_: Optional[np.ndarray] = None

    def compute_feature_importance(
        self,
        model: object,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_size: int = 1000,
    ) -> np.ndarray:
        """
        Compute feature importance using the specified strategy.

        Args:
            model: Trained model
            X: Features DataFrame
            y: Target Series (required for permutation importance)
            sample_size: Number of samples to use for SHAP computation

        Returns:
            Array of feature importance values
        """
        if self.strategy == "shap":
            return self._compute_shap_importance(model, X, sample_size)
        elif self.strategy == "permutation":
            return self._compute_permutation_importance(model, X, y)
        elif self.strategy == "gradient":
            return self._compute_gradient_importance(model, X)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _compute_shap_importance(
        self, model: object, X: pd.DataFrame, sample_size: int
    ) -> np.ndarray:
        """
        Compute SHAP-based feature importance.

        Args:
            model: Trained model
            X: Features DataFrame
            sample_size: Number of samples for SHAP computation

        Returns:
            Array of absolute mean SHAP values per feature
        """
        if not SHAP_AVAILABLE:
            logger.warning("shap not installed, falling back to model-based importance")
            return self._get_model_importance(model, X)
        try:
            # Sample data if too large
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X

            # For ensemble models, compute SHAP for each base model and average
            if hasattr(model, "models_") and isinstance(model.models_, dict):
                importances = []
                for model_name, base_model in model.models_.items():
                    try:
                        if model_name == "lightgbm":
                            explainer = shap.TreeExplainer(base_model)
                        else:
                            explainer = shap.TreeExplainer(base_model)

                        shap_values = explainer.shap_values(X_sample)

                        # Handle different SHAP value formats
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # For binary classification

                        importance = np.abs(shap_values).mean(axis=0)
                        importances.append(importance)
                    except Exception as e:
                        logger.debug(f"SHAP failed for {model_name}: {e}")
                        continue

                if importances:
                    importance = np.mean(importances, axis=0)
                    logger.debug(f"Computed ensemble SHAP importance for {len(importance)} features")
                    return importance
                else:
                    return self._get_model_importance(model, X)
            else:
                # Create SHAP explainer based on model type
                if hasattr(model, "predict_proba"):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_sample)

                # Compute SHAP values
                shap_values = explainer.shap_values(X_sample)

                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification

                # Compute mean absolute SHAP values
                importance = np.abs(shap_values).mean(axis=0)

                logger.debug(f"Computed SHAP importance for {len(importance)} features")
                return importance

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}, using model importance")
            return self._get_model_importance(model, X)

    def _compute_permutation_importance(
        self, model: object, X: pd.DataFrame, y: pd.Series
    ) -> np.ndarray:
        """
        Compute permutation-based feature importance.

        Args:
            model: Trained model
            X: Features DataFrame
            y: Target Series

        Returns:
            Array of permutation importance values
        """
        from sklearn.metrics import roc_auc_score

        baseline_pred = model.predict_proba(X)[:, 1]
        baseline_score = roc_auc_score(y, baseline_pred)

        importance = np.zeros(X.shape[1])

        for i, col in enumerate(X.columns):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            permuted_pred = model.predict_proba(X_permuted)[:, 1]
            permuted_score = roc_auc_score(y, permuted_pred)

            importance[i] = baseline_score - permuted_score

        logger.debug(f"Computed permutation importance for {len(importance)} features")
        return np.maximum(importance, 0)  # Only keep positive importance

    def _compute_gradient_importance(self, model: object, X: pd.DataFrame) -> np.ndarray:
        """
        Compute gradient-based feature importance approximation.

        Args:
            model: Trained model
            X: Features DataFrame

        Returns:
            Array of gradient-based importance values
        """
        # For tree-based models, use built-in feature importance
        return self._get_model_importance(model, X)

    def _get_model_importance(self, model: object, X: pd.DataFrame) -> np.ndarray:
        """
        Get model's native feature importance.

        Args:
            model: Trained model
            X: Features DataFrame

        Returns:
            Array of feature importance values
        """
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            return np.abs(model.coef_).flatten()
        else:
            logger.warning("Model has no feature importance, using uniform weights")
            return np.ones(X.shape[1]) / X.shape[1]

    def update_importance(
        self, model: object, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Update feature importance with temporal decay.

        Args:
            model: Trained model
            X: Features DataFrame
            y: Target Series

        Returns:
            Updated feature importance array
        """
        # Compute current importance
        current_importance = self.compute_feature_importance(model, X, y)

        # Apply temporal decay to historical importance
        if len(self.feature_importance_history_) > 0:
            decayed_history = [
                imp * (self.temporal_decay ** (len(self.feature_importance_history_) - i))
                for i, imp in enumerate(self.feature_importance_history_)
            ]
            historical_importance = np.mean(decayed_history, axis=0)

            # Combine current and historical
            self.current_importance_ = (
                0.5 * current_importance + 0.5 * historical_importance
            )
        else:
            self.current_importance_ = current_importance

        # Store in history
        self.feature_importance_history_.append(current_importance)

        # Keep only recent history (last 10 updates)
        if len(self.feature_importance_history_) > 10:
            self.feature_importance_history_.pop(0)

        return self.current_importance_

    def compute_sample_weights(
        self, X: pd.DataFrame, y: pd.Series, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute adaptive sample weights based on feature importance.

        Args:
            X: Features DataFrame
            y: Target Series
            predictions: Model predictions

        Returns:
            Array of sample weights
        """
        if self.current_importance_ is None:
            return np.ones(len(X))

        # Get top-k most important features
        top_k_indices = np.argsort(self.current_importance_)[-self.top_k_features :]

        # Compute sample-level importance scores
        X_array = X.values
        sample_importance = np.abs(X_array[:, top_k_indices]).sum(axis=1)

        # Normalize to [0, 1]
        sample_importance = (sample_importance - sample_importance.min()) / (
            sample_importance.max() - sample_importance.min() + 1e-8
        )

        # Compute prediction error
        prediction_error = np.abs(y.values - predictions)

        # Stratify by risk segments
        segment_weights = self._compute_segment_weights(predictions, y)

        # Combine importance, error, and segment weights
        weights = (
            1.0
            + self.reweight_alpha * sample_importance
            + self.reweight_alpha * prediction_error
            + self.reweight_alpha * segment_weights
        )

        # Normalize weights
        weights = weights / weights.mean()

        logger.debug(f"Computed sample weights: mean={weights.mean():.3f}, std={weights.std():.3f}")
        return weights

    def _compute_segment_weights(self, predictions: np.ndarray, y: pd.Series) -> np.ndarray:
        """
        Compute weights based on risk segments.

        Args:
            predictions: Model predictions
            y: Target Series

        Returns:
            Array of segment-based weights
        """
        # Divide predictions into risk segments
        segment_indices = np.digitize(predictions, np.linspace(0, 1, self.segment_bins + 1))

        # Compute segment-level error rates
        segment_weights = np.zeros(len(predictions))

        for segment in range(1, self.segment_bins + 1):
            mask = segment_indices == segment
            if mask.sum() > 0:
                segment_error = np.abs(y.values[mask] - predictions[mask]).mean()
                segment_weights[mask] = segment_error

        return segment_weights


class CurriculumScheduler:
    """
    Curriculum learning scheduler that progressively focuses on harder samples.

    Implements temporal curriculum learning where the model gradually transitions
    from easy to hard samples based on prediction difficulty.
    """

    def __init__(
        self,
        warmup_epochs: int = 15,
        initial_easy_ratio: float = 0.7,
        final_easy_ratio: float = 0.3,
        difficulty_metric: str = "prediction_variance",
        schedule: str = "linear",
    ):
        """
        Initialize curriculum scheduler.

        Args:
            warmup_epochs: Number of epochs before starting curriculum
            initial_easy_ratio: Initial proportion of easy samples
            final_easy_ratio: Final proportion of easy samples
            difficulty_metric: Metric for measuring difficulty
            schedule: Schedule type ('linear', 'cosine', 'exponential')
        """
        self.warmup_epochs = warmup_epochs
        self.initial_easy_ratio = initial_easy_ratio
        self.final_easy_ratio = final_easy_ratio
        self.difficulty_metric = difficulty_metric
        self.schedule = schedule

        self.current_epoch_ = 0
        self.difficulty_scores_: Optional[np.ndarray] = None

    def compute_difficulty(
        self, predictions: np.ndarray, y: pd.Series, prediction_history: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute difficulty scores for samples.

        Args:
            predictions: Current predictions
            y: Target values
            prediction_history: History of predictions for variance computation

        Returns:
            Array of difficulty scores (higher = harder)
        """
        if self.difficulty_metric == "prediction_variance":
            if len(prediction_history) > 1:
                # Compute variance across recent predictions
                recent_preds = np.array(prediction_history[-5:])  # Last 5 predictions
                difficulty = np.var(recent_preds, axis=0)
            else:
                # Use prediction uncertainty (distance from 0.5)
                difficulty = 1 - np.abs(predictions - 0.5) * 2
        elif self.difficulty_metric == "loss":
            # Binary cross-entropy loss
            epsilon = 1e-7
            difficulty = -(
                y.values * np.log(predictions + epsilon)
                + (1 - y.values) * np.log(1 - predictions + epsilon)
            )
        elif self.difficulty_metric == "uncertainty":
            # Entropy-based uncertainty
            epsilon = 1e-7
            difficulty = -(
                predictions * np.log(predictions + epsilon)
                + (1 - predictions) * np.log(1 - predictions + epsilon)
            )
        else:
            raise ValueError(f"Unknown difficulty metric: {self.difficulty_metric}")

        self.difficulty_scores_ = difficulty
        return difficulty

    def get_curriculum_weights(self, epoch: int) -> float:
        """
        Get the current easy sample ratio based on the schedule.

        Args:
            epoch: Current epoch number

        Returns:
            Current easy sample ratio
        """
        self.current_epoch_ = epoch

        if epoch < self.warmup_epochs:
            return self.initial_easy_ratio

        # Compute progress through curriculum
        progress = (epoch - self.warmup_epochs) / max(1, 100 - self.warmup_epochs)
        progress = min(progress, 1.0)

        if self.schedule == "linear":
            ratio = self.initial_easy_ratio + progress * (
                self.final_easy_ratio - self.initial_easy_ratio
            )
        elif self.schedule == "cosine":
            ratio = self.final_easy_ratio + 0.5 * (
                self.initial_easy_ratio - self.final_easy_ratio
            ) * (1 + np.cos(np.pi * progress))
        elif self.schedule == "exponential":
            ratio = self.final_easy_ratio + (
                self.initial_easy_ratio - self.final_easy_ratio
            ) * np.exp(-3 * progress)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return ratio

    def get_sample_mask(self, difficulty: np.ndarray, epoch: int) -> np.ndarray:
        """
        Get boolean mask for samples to include based on curriculum.

        Args:
            difficulty: Array of difficulty scores
            epoch: Current epoch

        Returns:
            Boolean mask indicating which samples to include
        """
        easy_ratio = self.get_curriculum_weights(epoch)

        # Rank samples by difficulty (lower rank = easier)
        difficulty_ranks = rankdata(difficulty, method="ordinal")
        threshold_rank = int(len(difficulty) * easy_ratio)

        # Ensure minimum threshold to avoid filtering out all samples
        threshold_rank = max(threshold_rank, 100)

        # Include samples below threshold (easier samples)
        mask = difficulty_ranks <= threshold_rank

        logger.debug(
            f"Curriculum epoch {epoch}: including {mask.sum()}/{len(mask)} samples "
            f"(easy_ratio={easy_ratio:.2f})"
        )

        return mask
