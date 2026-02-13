"""
Adaptive Feature Importance Reweighting for Credit Risk Prediction.

This package implements a novel approach to credit default prediction using
dynamic feature importance reweighting that adapts during training.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_feature_importance_reweighting_credit_risk.models.model import (
    AdaptiveEnsembleModel,
)
from adaptive_feature_importance_reweighting_credit_risk.training.trainer import (
    AdaptiveTrainer,
)

__all__ = [
    "AdaptiveEnsembleModel",
    "AdaptiveTrainer",
]
