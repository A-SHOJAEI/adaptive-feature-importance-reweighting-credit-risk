"""Training loop with adaptive feature importance reweighting and curriculum learning."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from adaptive_feature_importance_reweighting_credit_risk.models.components import (
    AdaptiveFeatureReweighter,
    CurriculumScheduler,
)
from adaptive_feature_importance_reweighting_credit_risk.models.model import (
    AdaptiveEnsembleModel,
)

logger = logging.getLogger(__name__)


class AdaptiveTrainer:
    """
    Trainer for adaptive feature importance reweighting with curriculum learning.

    Implements iterative training with progressive feature importance updates
    and curriculum learning schedule.
    """

    def __init__(
        self,
        model: AdaptiveEnsembleModel,
        reweighting_config: Optional[Dict[str, Any]] = None,
        curriculum_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "models",
        monitor_metric: str = "roc_auc",
        mode: str = "max",
    ):
        """
        Initialize adaptive trainer.

        Args:
            model: Ensemble model to train
            reweighting_config: Configuration for adaptive reweighting
            curriculum_config: Configuration for curriculum learning
            training_config: Training configuration
            checkpoint_dir: Directory to save checkpoints
            monitor_metric: Metric to monitor for best model
            mode: 'max' or 'min' for metric optimization
        """
        self.model = model
        self.reweighting_config = reweighting_config or {}
        self.curriculum_config = curriculum_config or {}
        self.training_config = training_config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.reweighter: Optional[AdaptiveFeatureReweighter] = None
        self.curriculum: Optional[CurriculumScheduler] = None

        if self.reweighting_config.get("enabled", False):
            self.reweighter = AdaptiveFeatureReweighter(
                strategy=self.reweighting_config.get("strategy", "shap"),
                top_k_features=self.reweighting_config.get("top_k_features", 20),
                reweight_alpha=self.reweighting_config.get("reweight_alpha", 0.3),
                temporal_decay=self.reweighting_config.get("temporal_decay", 0.95),
                segment_bins=self.reweighting_config.get("segment_bins", 5),
                min_importance_threshold=self.reweighting_config.get("min_importance_threshold", 0.01),
            )

        if self.curriculum_config.get("enabled", False):
            self.curriculum = CurriculumScheduler(
                warmup_epochs=self.curriculum_config.get("warmup_epochs", 15),
                initial_easy_ratio=self.curriculum_config.get("initial_easy_ratio", 0.7),
                final_easy_ratio=self.curriculum_config.get("final_easy_ratio", 0.3),
                difficulty_metric=self.curriculum_config.get("difficulty_metric", "prediction_variance"),
                schedule=self.curriculum_config.get("schedule", "linear"),
            )

        # Training state
        self.best_score_: Optional[float] = None
        self.best_epoch_: int = 0
        self.history_: Dict[str, List[float]] = {}
        self.prediction_history_: List[np.ndarray] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_boost_round: int = 100,
        early_stopping_rounds: Optional[int] = 20,
        verbose: int = 10,
    ) -> Dict[str, Any]:
        """
        Train model with adaptive reweighting and curriculum learning.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            num_boost_round: Number of boosting rounds (epochs)
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity interval

        Returns:
            Dictionary with training history and results
        """
        logger.info(f"Starting training for {num_boost_round} rounds")
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

        best_score = -np.inf if self.mode == "max" else np.inf
        patience_counter = 0

        # Initialize history
        self.history_ = {"train_loss": [], "val_loss": [], "val_metric": []}

        for epoch in tqdm(range(num_boost_round), desc="Training"):
            # Get sample weights and curriculum mask
            sample_weight = self._get_sample_weights(
                X_train, y_train, epoch
            )

            curriculum_mask = self._get_curriculum_mask(
                X_train, y_train, epoch
            )

            # Apply curriculum mask
            if curriculum_mask is not None:
                X_train_epoch = X_train[curriculum_mask]
                y_train_epoch = y_train[curriculum_mask]
                sample_weight_epoch = sample_weight[curriculum_mask] if sample_weight is not None else None
            else:
                X_train_epoch = X_train
                y_train_epoch = y_train
                sample_weight_epoch = sample_weight

            # Train model for one round
            self.model.fit(
                X_train_epoch,
                y_train_epoch,
                sample_weight=sample_weight_epoch,
                eval_set=[(X_val, y_val)],
                verbose=0,
            )

            # Update feature importance
            if self.reweighter is not None and epoch % self.reweighting_config.get("update_frequency", 5) == 0:
                if epoch >= self.reweighting_config.get("warmup_epochs", 10):
                    self.reweighter.update_importance(self.model, X_train, y_train)

            # Evaluate
            val_pred = self.model.predict_proba(X_val)[:, 1]
            self.prediction_history_.append(val_pred)

            # Compute metrics (placeholder - will be computed by evaluation module)
            from sklearn.metrics import roc_auc_score
            val_score = roc_auc_score(y_val, val_pred)

            self.history_["val_metric"].append(val_score)

            # Check for improvement
            is_improvement = (
                (self.mode == "max" and val_score > best_score)
                or (self.mode == "min" and val_score < best_score)
            )

            if is_improvement:
                best_score = val_score
                self.best_score_ = best_score
                self.best_epoch_ = epoch
                patience_counter = 0

                # Save best model
                self.save_checkpoint("best_model.pkl")

                if verbose > 0 and epoch % verbose == 0:
                    logger.info(
                        f"Epoch {epoch}: val_{self.monitor_metric}={val_score:.4f} (best)"
                    )
            else:
                patience_counter += 1

                if verbose > 0 and epoch % verbose == 0:
                    logger.info(
                        f"Epoch {epoch}: val_{self.monitor_metric}={val_score:.4f}"
                    )

            # Early stopping
            if early_stopping_rounds is not None and patience_counter >= early_stopping_rounds:
                logger.info(
                    f"Early stopping at epoch {epoch}, "
                    f"best {self.monitor_metric}={best_score:.4f} at epoch {self.best_epoch_}"
                )
                break

        logger.info(f"Training completed. Best {self.monitor_metric}={best_score:.4f}")

        return {
            "best_score": best_score,
            "best_epoch": self.best_epoch_,
            "history": self.history_,
        }

    def _get_sample_weights(
        self, X: pd.DataFrame, y: pd.Series, epoch: int
    ) -> Optional[np.ndarray]:
        """
        Compute sample weights for current epoch.

        Args:
            X: Features
            y: Target
            epoch: Current epoch

        Returns:
            Sample weights or None
        """
        if self.reweighter is None:
            return None

        if epoch < self.reweighting_config.get("warmup_epochs", 10):
            return None

        # Get predictions
        predictions = self.model.predict_proba(X)[:, 1]

        # Compute adaptive weights
        weights = self.reweighter.compute_sample_weights(X, y, predictions)

        return weights

    def _get_curriculum_mask(
        self, X: pd.DataFrame, y: pd.Series, epoch: int
    ) -> Optional[np.ndarray]:
        """
        Get curriculum mask for current epoch.

        Args:
            X: Features
            y: Target
            epoch: Current epoch

        Returns:
            Boolean mask or None
        """
        if self.curriculum is None:
            return None

        if epoch < self.curriculum_config.get("warmup_epochs", 15):
            return None

        # Get predictions
        predictions = self.model.predict_proba(X)[:, 1]

        # Compute difficulty
        difficulty = self.curriculum.compute_difficulty(
            predictions, y, self.prediction_history_
        )

        # Get curriculum mask
        mask = self.curriculum.get_sample_mask(difficulty, epoch)

        return mask

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model": self.model,
            "reweighter": self.reweighter,
            "curriculum": self.curriculum,
            "best_score": self.best_score_,
            "best_epoch": self.best_epoch_,
            "history": self.history_,
        }

        joblib.dump(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = joblib.load(checkpoint_path)

        self.model = checkpoint["model"]
        self.reweighter = checkpoint.get("reweighter")
        self.curriculum = checkpoint.get("curriculum")
        self.best_score_ = checkpoint.get("best_score")
        self.best_epoch_ = checkpoint.get("best_epoch")
        self.history_ = checkpoint.get("history", {})

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
