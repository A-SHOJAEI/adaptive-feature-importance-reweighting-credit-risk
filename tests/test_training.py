"""Tests for training functionality."""

import numpy as np
import pandas as pd
import pytest

from adaptive_feature_importance_reweighting_credit_risk.evaluation.metrics import (
    compute_ks_statistic,
    compute_metrics,
    compute_optimal_threshold,
)
from adaptive_feature_importance_reweighting_credit_risk.models.model import (
    AdaptiveEnsembleModel,
)
from adaptive_feature_importance_reweighting_credit_risk.training.trainer import (
    AdaptiveTrainer,
)


class TestAdaptiveTrainer:
    """Tests for adaptive trainer."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = AdaptiveEnsembleModel(model_type="lightgbm")

        trainer = AdaptiveTrainer(
            model=model,
            reweighting_config={"enabled": False},
            curriculum_config={"enabled": False},
        )

        assert trainer.model is model
        assert trainer.reweighter is None
        assert trainer.curriculum is None

    def test_trainer_with_reweighting(self):
        """Test trainer with reweighting enabled."""
        model = AdaptiveEnsembleModel(model_type="lightgbm")

        reweighting_config = {
            "enabled": True,
            "strategy": "gradient",
            "warmup_epochs": 2,
            "update_frequency": 2,
        }

        trainer = AdaptiveTrainer(model=model, reweighting_config=reweighting_config)

        assert trainer.reweighter is not None
        assert trainer.reweighter.strategy == "gradient"

    def test_trainer_with_curriculum(self):
        """Test trainer with curriculum learning enabled."""
        model = AdaptiveEnsembleModel(model_type="lightgbm")

        curriculum_config = {
            "enabled": True,
            "warmup_epochs": 2,
            "schedule": "linear",
        }

        trainer = AdaptiveTrainer(model=model, curriculum_config=curriculum_config)

        assert trainer.curriculum is not None
        assert trainer.curriculum.schedule == "linear"

    def test_trainer_train(self, sample_train_val_data, temp_output_dir):
        """Test basic training."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        trainer = AdaptiveTrainer(
            model=model,
            checkpoint_dir=str(temp_output_dir),
            monitor_metric="roc_auc",
            mode="max",
        )

        results = trainer.train(
            X_train,
            y_train,
            X_val,
            y_val,
            num_boost_round=3,
            early_stopping_rounds=2,
            verbose=0,
        )

        assert "best_score" in results
        assert "best_epoch" in results
        assert "history" in results
        assert trainer.best_score_ is not None

    def test_trainer_save_checkpoint(self, sample_train_val_data, temp_output_dir):
        """Test checkpoint saving."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        trainer = AdaptiveTrainer(model=model, checkpoint_dir=str(temp_output_dir))

        trainer.train(X_train, y_train, X_val, y_val, num_boost_round=2, verbose=0)

        checkpoint_path = temp_output_dir / "best_model.pkl"
        assert checkpoint_path.exists()

    def test_trainer_load_checkpoint(self, sample_train_val_data, temp_output_dir):
        """Test checkpoint loading."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        trainer = AdaptiveTrainer(model=model, checkpoint_dir=str(temp_output_dir))

        trainer.train(X_train, y_train, X_val, y_val, num_boost_round=2, verbose=0)

        # Load checkpoint
        new_trainer = AdaptiveTrainer(
            model=AdaptiveEnsembleModel(model_type="lightgbm"),
            checkpoint_dir=str(temp_output_dir),
        )

        new_trainer.load_checkpoint("best_model.pkl")

        assert new_trainer.model is not None
        assert new_trainer.best_score_ is not None


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_metrics(self):
        """Test metric computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.85])

        metrics = compute_metrics(y_true, y_pred_proba)

        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "ks_statistic" in metrics
        assert "gini" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check value ranges
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_compute_ks_statistic(self):
        """Test KS statistic computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.85])

        ks_stat = compute_ks_statistic(y_true, y_pred_proba)

        assert isinstance(ks_stat, (float, np.floating))
        assert 0 <= ks_stat <= 1

    def test_compute_optimal_threshold(self):
        """Test optimal threshold computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.85])

        threshold = compute_optimal_threshold(y_true, y_pred_proba, metric="f1")

        assert isinstance(threshold, (float, np.floating))
        assert 0 <= threshold <= 1

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

        metrics = compute_metrics(y_true, y_pred_proba)

        assert metrics["roc_auc"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_metrics_with_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.uniform(0, 1, 100)

        metrics = compute_metrics(y_true, y_pred_proba)

        # For random predictions, AUC should be around 0.5
        assert 0.3 <= metrics["roc_auc"] <= 0.7
