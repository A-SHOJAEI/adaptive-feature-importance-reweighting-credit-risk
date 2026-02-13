"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    from sklearn.metrics import average_precision_score

    ap = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PR curve to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None,
    title: str = "Feature Importance",
) -> None:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        output_path: Path to save plot
        title: Plot title
    """
    # Sort and select top N
    top_features = importance_df.nlargest(top_n, "importance")

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["importance"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature importance to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Calibration Curve",
) -> None:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        output_path: Path to save plot
        title: Plot title
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute mean predicted and true probabilities per bin
    bin_sums = np.bincount(bin_indices, weights=y_pred_proba, minlength=n_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Avoid division by zero
    nonzero = bin_counts > 0
    mean_predicted = np.zeros(n_bins)
    mean_true = np.zeros(n_bins)
    mean_predicted[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    mean_true[nonzero] = bin_true[nonzero] / bin_counts[nonzero]

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(mean_predicted, mean_true, "o-", label="Model", linewidth=2, markersize=8)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved calibration curve to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, list],
    output_path: Optional[str] = None,
    title: str = "Training History",
) -> None:
    """
    Plot training history.

    Args:
        history: Dictionary with metric names and values per epoch
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    for metric_name, values in history.items():
        if len(values) > 0:
            plt.plot(values, label=metric_name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training history to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_lift_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Lift Curve",
) -> None:
    """
    Plot lift curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        output_path: Path to save plot
        title: Plot title
    """
    from adaptive_feature_importance_reweighting_credit_risk.evaluation.metrics import (
        compute_lift_curve,
    )

    lift_df = compute_lift_curve(y_true, y_pred_proba, n_bins)

    plt.figure(figsize=(10, 6))
    plt.plot(lift_df["percentile"], lift_df["lift"], "o-", linewidth=2, markersize=8)
    plt.axhline(y=1, color="k", linestyle="--", label="Baseline")
    plt.xlabel("Percentile (%)")
    plt.ylabel("Lift")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved lift curve to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: Optional[pd.DataFrame] = None,
    output_dir: str = "results",
) -> None:
    """
    Generate comprehensive evaluation report with all plots.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels
        feature_importance: Feature importance DataFrame
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation report...")

    # ROC curve
    plot_roc_curve(y_true, y_pred_proba, output_path / "roc_curve.png")

    # Precision-recall curve
    plot_precision_recall_curve(y_true, y_pred_proba, output_path / "pr_curve.png")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_path / "confusion_matrix.png")

    # Calibration curve
    plot_calibration_curve(y_true, y_pred_proba, output_path / "calibration_curve.png")

    # Lift curve
    plot_lift_curve(y_true, y_pred_proba, output_path / "lift_curve.png")

    # Feature importance
    if feature_importance is not None:
        plot_feature_importance(feature_importance, output_path=output_path / "feature_importance.png")

    logger.info(f"Evaluation report saved to {output_dir}")
