"""Evaluation metrics for credit risk prediction."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for credit risk prediction.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels (computed from y_pred_proba if not provided)
        threshold: Classification threshold

    Returns:
        Dictionary of metric names and values
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}

    # ROC AUC
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC: {e}")
        metrics["roc_auc"] = 0.0

    # PR AUC (Average Precision)
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not compute PR AUC: {e}")
        metrics["pr_auc"] = 0.0

    # KS Statistic
    try:
        metrics["ks_statistic"] = compute_ks_statistic(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not compute KS statistic: {e}")
        metrics["ks_statistic"] = 0.0

    # Gini coefficient
    try:
        metrics["gini"] = 2 * metrics["roc_auc"] - 1
    except Exception as e:
        logger.warning(f"Could not compute Gini: {e}")
        metrics["gini"] = 0.0

    # Brier score
    try:
        metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not compute Brier score: {e}")
        metrics["brier_score"] = 0.0

    # Classification metrics
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        logger.warning(f"Could not compute classification metrics: {e}")
        metrics["accuracy"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0

    # Confusion matrix derived metrics
    try:
        tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
        metrics["true_positive_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["true_negative_rate"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    except Exception as e:
        logger.warning(f"Could not compute confusion matrix metrics: {e}")

    return metrics


def compute_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov statistic.

    The KS statistic measures the maximum separation between cumulative
    distributions of positive and negative classes.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        KS statistic value
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks_statistic = np.max(tpr - fpr)
    return ks_statistic


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Compute confusion matrix components.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Tuple of (true_negatives, false_positives, false_negatives, true_positives)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tn, fp, fn, tp


def compute_metrics_by_segment(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_segments: int = 5,
) -> pd.DataFrame:
    """
    Compute metrics by risk segments (deciles/quantiles).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_segments: Number of segments to create

    Returns:
        DataFrame with metrics per segment
    """
    # Create segments based on predicted probabilities
    segment_labels = pd.qcut(
        y_pred_proba,
        q=n_segments,
        labels=[f"Segment_{i+1}" for i in range(n_segments)],
        duplicates="drop",
    )

    results = []

    for segment in segment_labels.unique():
        mask = segment_labels == segment
        segment_y_true = y_true[mask]
        segment_y_pred_proba = y_pred_proba[mask]

        segment_metrics = {
            "segment": segment,
            "count": mask.sum(),
            "default_rate": segment_y_true.mean(),
            "mean_pred_proba": segment_y_pred_proba.mean(),
            "min_pred_proba": segment_y_pred_proba.min(),
            "max_pred_proba": segment_y_pred_proba.max(),
        }

        results.append(segment_metrics)

    return pd.DataFrame(results)


def compute_lift_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute lift curve data.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for lift calculation

    Returns:
        DataFrame with lift curve data
    """
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Compute cumulative metrics
    n_samples = len(y_true)
    bin_size = n_samples // n_bins

    results = []

    for i in range(1, n_bins + 1):
        end_idx = min(i * bin_size, n_samples)
        bin_y_true = y_true_sorted[:end_idx]

        results.append(
            {
                "percentile": i * (100 / n_bins),
                "samples": end_idx,
                "positives_captured": bin_y_true.sum(),
                "capture_rate": bin_y_true.sum() / y_true.sum() if y_true.sum() > 0 else 0,
                "lift": (bin_y_true.sum() / end_idx) / (y_true.sum() / n_samples)
                if y_true.sum() > 0
                else 0,
            }
        )

    return pd.DataFrame(results)


def compute_optimal_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = "f1"
) -> float:
    """
    Compute optimal classification threshold.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')

    Returns:
        Optimal threshold value
    """
    if metric == "youden":
        # Youden's J statistic: sensitivity + specificity - 1
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    # For other metrics, search over threshold range
    thresholds = np.linspace(0, 1, 101)
    best_score = -np.inf
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold
