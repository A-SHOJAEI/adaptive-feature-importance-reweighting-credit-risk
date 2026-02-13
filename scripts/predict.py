#!/usr/bin/env python
"""Prediction script for inference on new data."""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_importance_reweighting_credit_risk.data.preprocessing import (
    CreditDataPreprocessor,
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
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pkl",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file (CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--include-proba",
        action="store_true",
        default=True,
        help="Include probability scores in output",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str) -> object:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Trained model
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = joblib.load(checkpoint_path)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    else:
        return checkpoint


def load_input_data(input_path: str) -> pd.DataFrame:
    """
    Load input data from file.

    Args:
        input_path: Path to input data file

    Returns:
        DataFrame with input features
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load based on file extension
    input_file = Path(input_path)

    if input_file.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_file.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_file.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    # Remove target column if present
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    return df


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input features.

    Args:
        X: Input features DataFrame

    Returns:
        Preprocessed features
    """
    preprocessor = CreditDataPreprocessor(
        scale_features=True,
        handle_missing="median",
        encode_categoricals=True,
        create_interactions=True,
    )

    # Fit and transform (in production, preprocessor should be saved with model)
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting Prediction")
    logger.info("=" * 80)

    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = load_checkpoint(args.checkpoint)
        logger.info("Model loaded successfully")

        # Load input data
        logger.info(f"Loading input data from {args.input}")
        X = load_input_data(args.input)
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")

        # Preprocess features
        logger.info("Preprocessing features...")
        X_preprocessed = preprocess_features(X)
        logger.info(f"Preprocessed to {len(X_preprocessed.columns)} features")

        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_proba = model.predict_proba(X_preprocessed)[:, 1]
        y_pred = (y_pred_proba >= args.threshold).astype(int)

        # Create output DataFrame
        results = pd.DataFrame({"predicted_label": y_pred})

        if args.include_proba:
            results["predicted_probability"] = y_pred_proba
            results["default_risk"] = pd.cut(
                y_pred_proba,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
            )

        # Add input index if present
        if "SK_ID_CURR" in X.columns:
            results.insert(0, "SK_ID_CURR", X["SK_ID_CURR"].values)
        elif X.index.name:
            results.insert(0, X.index.name, X.index.values)

        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Prediction Summary")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(results)}")
        logger.info(f"Predicted defaults: {y_pred.sum()} ({y_pred.mean()*100:.2f}%)")
        logger.info(f"Predicted no defaults: {(1-y_pred).sum()} ({(1-y_pred.mean())*100:.2f}%)")

        if args.include_proba:
            logger.info(f"\nMean predicted probability: {y_pred_proba.mean():.4f}")
            logger.info(f"Std predicted probability: {y_pred_proba.std():.4f}")
            logger.info(f"Min predicted probability: {y_pred_proba.min():.4f}")
            logger.info(f"Max predicted probability: {y_pred_proba.max():.4f}")

            logger.info("\nRisk distribution:")
            risk_counts = results["default_risk"].value_counts().sort_index()
            for risk_level, count in risk_counts.items():
                logger.info(f"  {risk_level}: {count} ({count/len(results)*100:.1f}%)")

        # Display first few predictions
        logger.info("\n" + "=" * 80)
        logger.info("Sample Predictions (first 10)")
        logger.info("=" * 80)
        print("\n" + results.head(10).to_string(index=False))

        logger.info("\n" + "=" * 80)
        logger.info("Prediction completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
