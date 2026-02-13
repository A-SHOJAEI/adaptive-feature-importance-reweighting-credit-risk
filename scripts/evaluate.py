#!/usr/bin/env python
"""Evaluation script for trained models."""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_importance_reweighting_credit_risk.data.loader import (
    load_dataset,
    split_data,
)
from adaptive_feature_importance_reweighting_credit_risk.data.preprocessing import (
    CreditDataPreprocessor,
)
from adaptive_feature_importance_reweighting_credit_risk.evaluation.analysis import (
    generate_evaluation_report,
)
from adaptive_feature_importance_reweighting_credit_risk.evaluation.metrics import (
    compute_metrics,
    compute_metrics_by_segment,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pkl",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (optional, will generate synthetic if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate evaluation plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing model and metadata
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = joblib.load(checkpoint_path)
    return checkpoint


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 80)

    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = load_checkpoint(args.checkpoint)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
            logger.info(f"Loaded checkpoint with best score: {checkpoint.get('best_score', 'N/A')}")
        else:
            model = checkpoint
            logger.info("Loaded model checkpoint")

        # Load or generate data
        logger.info("Loading evaluation data...")
        if args.data and Path(args.data).exists():
            # Load from file
            df = pd.read_csv(args.data)
            if "TARGET" in df.columns:
                y = df["TARGET"]
                X = df.drop(columns=["TARGET"])
            else:
                raise ValueError("Data file must contain 'TARGET' column")
        else:
            # Generate synthetic data
            X, y = load_dataset(
                dataset_name="home_credit",
                sample_frac=0.3,  # Use subset for evaluation
                min_samples=10000,
                random_state=args.seed,
            )

            # Use test split
            splits = split_data(
                X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=args.seed
            )
            X, y = splits["test"]

        logger.info(f"Loaded {len(X)} samples for evaluation")

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = CreditDataPreprocessor(
            scale_features=True,
            handle_missing="median",
            encode_categoricals=True,
            create_interactions=True,
        )

        # Fit on full data (in production, this would be saved with the model)
        X_preprocessed = preprocessor.fit_transform(X)

        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_proba = model.predict_proba(X_preprocessed)[:, 1]
        y_pred = model.predict(X_preprocessed)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_metrics(y.values, y_pred_proba, y_pred)

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Metrics")
        logger.info("=" * 80)

        # Print metrics in a formatted table
        print("\n{:<30s} {:>10s}".format("Metric", "Value"))
        print("-" * 42)
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:<30s} {value:>10.4f}")

        # Compute metrics by segment
        logger.info("\nComputing segment analysis...")
        segment_metrics = compute_metrics_by_segment(y.values, y_pred_proba, n_segments=5)

        logger.info("\n" + "=" * 80)
        logger.info("Metrics by Risk Segment")
        logger.info("=" * 80)
        print("\n" + segment_metrics.to_string(index=False))

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "checkpoint": args.checkpoint,
            "n_samples": len(X),
            "metrics": metrics,
            "segment_metrics": segment_metrics.to_dict(orient="records"),
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

        # Save predictions
        predictions_df = pd.DataFrame(
            {
                "true_label": y.values,
                "predicted_proba": y_pred_proba,
                "predicted_label": y_pred,
            }
        )

        predictions_file = Path(args.output_dir) / "predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")

        # Generate plots
        if args.generate_plots:
            logger.info("\nGenerating evaluation plots...")
            try:
                # Get feature importance if available
                feature_importance = None
                if hasattr(model, "get_feature_importance_df"):
                    feature_importance = model.get_feature_importance_df()

                generate_evaluation_report(
                    y.values,
                    y_pred_proba,
                    y_pred,
                    feature_importance,
                    output_dir=args.output_dir,
                )
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
