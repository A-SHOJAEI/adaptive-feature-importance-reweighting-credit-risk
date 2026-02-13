"""Tests for model implementations."""

import numpy as np
import pandas as pd
import pytest

from adaptive_feature_importance_reweighting_credit_risk.models.components import (
    AdaptiveFeatureReweighter,
    CurriculumScheduler,
)
from adaptive_feature_importance_reweighting_credit_risk.models.model import (
    AdaptiveEnsembleModel,
)


class TestAdaptiveEnsembleModel:
    """Tests for adaptive ensemble model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = AdaptiveEnsembleModel(model_type="lightgbm")

        assert model.model_type == "lightgbm"
        assert model.models_ == {}

    def test_model_fit(self, sample_train_val_data):
        """Test model fitting."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert "lightgbm" in model.models_
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == X_train.shape[1]

    def test_model_predict_proba(self, sample_train_val_data):
        """Test probability prediction."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)

        assert y_pred_proba.shape == (len(X_val), 2)
        assert np.all(y_pred_proba >= 0)
        assert np.all(y_pred_proba <= 1)
        assert np.allclose(y_pred_proba.sum(axis=1), 1.0)

    def test_model_predict(self, sample_train_val_data):
        """Test class prediction."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        assert y_pred.shape == (len(X_val),)
        assert set(y_pred).issubset({0, 1})

    def test_model_feature_importance(self, sample_train_val_data):
        """Test feature importance computation."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )

        model.fit(X_train, y_train)
        importance_df = model.get_feature_importance_df()

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == X_train.shape[1]

    def test_ensemble_model(self, sample_train_val_data):
        """Test ensemble with multiple models."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="ensemble",
            lightgbm_params={"n_estimators": 5, "random_state": 42, "verbose": -1},
            xgboost_params={"n_estimators": 5, "random_state": 42, "verbosity": 0},
            catboost_params={"iterations": 5, "random_state": 42, "verbose": 0},
        )

        model.fit(X_train, y_train)

        # Check all models were trained
        assert "lightgbm" in model.models_
        assert "xgboost" in model.models_
        assert "catboost" in model.models_

        # Check predictions work
        y_pred_proba = model.predict_proba(X_val)
        assert y_pred_proba.shape == (len(X_val), 2)


class TestAdaptiveFeatureReweighter:
    """Tests for adaptive feature reweighter."""

    def test_reweighter_initialization(self):
        """Test reweighter initialization."""
        reweighter = AdaptiveFeatureReweighter(
            strategy="shap", top_k_features=10, reweight_alpha=0.3
        )

        assert reweighter.strategy == "shap"
        assert reweighter.top_k_features == 10
        assert reweighter.reweight_alpha == 0.3

    def test_compute_feature_importance(self, sample_train_val_data):
        """Test feature importance computation."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        # Train a simple model
        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )
        model.fit(X_train, y_train)

        reweighter = AdaptiveFeatureReweighter(strategy="gradient")
        importance = reweighter.compute_feature_importance(model, X_train, y_train)

        assert isinstance(importance, np.ndarray)
        assert len(importance) == X_train.shape[1]
        assert np.all(importance >= 0)

    def test_update_importance(self, sample_train_val_data):
        """Test importance update with temporal decay."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )
        model.fit(X_train, y_train)

        reweighter = AdaptiveFeatureReweighter(temporal_decay=0.9)

        # Update multiple times
        importance_1 = reweighter.update_importance(model, X_train, y_train)
        importance_2 = reweighter.update_importance(model, X_train, y_train)

        assert len(reweighter.feature_importance_history_) == 2
        assert reweighter.current_importance_ is not None

    def test_compute_sample_weights(self, sample_train_val_data):
        """Test sample weight computation."""
        X_train, y_train, X_val, y_val = sample_train_val_data

        model = AdaptiveEnsembleModel(
            model_type="lightgbm",
            lightgbm_params={"n_estimators": 10, "random_state": 42, "verbose": -1},
        )
        model.fit(X_train, y_train)

        reweighter = AdaptiveFeatureReweighter()
        reweighter.update_importance(model, X_train, y_train)

        predictions = model.predict_proba(X_train)[:, 1]
        weights = reweighter.compute_sample_weights(X_train, y_train, predictions)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(X_train)
        assert np.all(weights > 0)


class TestCurriculumScheduler:
    """Tests for curriculum scheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = CurriculumScheduler(
            warmup_epochs=10, initial_easy_ratio=0.7, final_easy_ratio=0.3
        )

        assert scheduler.warmup_epochs == 10
        assert scheduler.initial_easy_ratio == 0.7
        assert scheduler.final_easy_ratio == 0.3

    def test_compute_difficulty(self, sample_data):
        """Test difficulty computation."""
        X, y = sample_data

        predictions = np.random.uniform(0, 1, len(y))
        prediction_history = [predictions]

        scheduler = CurriculumScheduler(difficulty_metric="prediction_variance")
        difficulty = scheduler.compute_difficulty(predictions, y, prediction_history)

        assert isinstance(difficulty, np.ndarray)
        assert len(difficulty) == len(y)
        assert np.all(difficulty >= 0)

    def test_get_curriculum_weights(self):
        """Test curriculum weight schedule."""
        scheduler = CurriculumScheduler(
            warmup_epochs=10,
            initial_easy_ratio=0.7,
            final_easy_ratio=0.3,
            schedule="linear",
        )

        # Before warmup
        ratio_0 = scheduler.get_curriculum_weights(0)
        assert ratio_0 == 0.7

        # After warmup
        ratio_50 = scheduler.get_curriculum_weights(50)
        assert 0.3 <= ratio_50 <= 0.7

        # Far into training
        ratio_100 = scheduler.get_curriculum_weights(100)
        assert abs(ratio_100 - 0.3) < 0.1

    def test_get_sample_mask(self, sample_data):
        """Test sample mask generation."""
        X, y = sample_data

        difficulty = np.random.uniform(0, 1, len(y))
        scheduler = CurriculumScheduler()

        mask = scheduler.get_sample_mask(difficulty, epoch=20)

        assert isinstance(mask, np.ndarray)
        assert len(mask) == len(y)
        assert mask.dtype == bool
        assert mask.sum() <= len(y)
