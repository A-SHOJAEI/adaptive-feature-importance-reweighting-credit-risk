#!/usr/bin/env python
"""Training script for adaptive feature importance reweighting."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

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
from adaptive_feature_importance_reweighting_credit_risk.evaluation.metrics import compute_metrics
from adaptive_feature_importance_reweighting_credit_risk.models.model import (
    AdaptiveEnsembleModel,
)
from adaptive_feature_importance_reweighting_credit_risk.training.trainer import (
    AdaptiveTrainer,
)
from adaptive_feature_importance_reweighting_credit_risk.utils.config import load_config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "train.log"

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train adaptive feature importance reweighting model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (not used for tree models, for compatibility)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["num_boost_round"] = args.epochs

    # Setup
    seed = config.get("seed", 42)
    set_seed(seed)

    log_level = config.get("logging", {}).get("level", "INFO")
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    setup_logging(log_dir, log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting Adaptive Feature Importance Reweighting Training")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Random seed: {seed}")

    # Setup MLflow (optional)
    mlflow_config = config.get("logging", {}).get("mlflow", {})
    mlflow_enabled = mlflow_config.get("enabled", False)

    if mlflow_enabled:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "file:./mlruns"))
            mlflow.set_experiment(mlflow_config.get("experiment_name", "adaptive_feature_importance"))
            mlflow.start_run()
            mlflow.log_params({"config": args.config, "seed": seed})
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
            mlflow_enabled = False

    try:
        # Load data
        logger.info("Loading dataset...")
        data_config = config.get("data", {})
        X, y = load_dataset(
            dataset_name=data_config.get("dataset_name", "home_credit"),
            sample_frac=data_config.get("sample_frac", 1.0),
            min_samples=data_config.get("min_samples", 10000),
            random_state=seed,
        )

        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        # Split data
        logger.info("Splitting data...")
        splits = split_data(
            X,
            y,
            train_size=data_config.get("train_size", 0.7),
            val_size=data_config.get("val_size", 0.15),
            test_size=data_config.get("test_size", 0.15),
            random_state=seed,
        )

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = CreditDataPreprocessor(
            scale_features=True,
            handle_missing="median",
            encode_categoricals=True,
            create_interactions=True,
        )

        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        logger.info(f"Preprocessed features: {len(X_train.columns)}")

        # Create model
        logger.info("Creating model...")
        model_config = config.get("model", {})
        model = AdaptiveEnsembleModel(
            model_type=model_config.get("type", "ensemble"),
            lightgbm_params=model_config.get("lightgbm", {}),
            xgboost_params=model_config.get("xgboost", {}),
            catboost_params=model_config.get("catboost", {}),
            ensemble_weights=model_config.get("ensemble_weights", {}),
        )

        # Create trainer
        logger.info("Creating trainer...")
        checkpoint_dir = config.get("checkpoint", {}).get("save_dir", "models")
        trainer = AdaptiveTrainer(
            model=model,
            reweighting_config=config.get("reweighting", {}),
            curriculum_config=config.get("curriculum", {}),
            training_config=config.get("training", {}),
            checkpoint_dir=checkpoint_dir,
            monitor_metric=config.get("checkpoint", {}).get("monitor", "roc_auc"),
            mode=config.get("checkpoint", {}).get("mode", "max"),
        )

        # Train model
        logger.info("Training model...")
        training_config = config.get("training", {})
        training_results = trainer.train(
            X_train,
            y_train,
            X_val,
            y_val,
            num_boost_round=training_config.get("num_boost_round", 100),
            early_stopping_rounds=training_config.get("early_stopping_rounds", 20),
            verbose=training_config.get("verbose_eval", 10),
        )

        logger.info(f"Training completed. Best score: {training_results['best_score']:.4f}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)

        test_metrics = compute_metrics(y_test.values, y_test_pred_proba, y_test_pred)

        logger.info("Test Set Metrics:")
        logger.info("-" * 40)
        for metric_name, value in test_metrics.items():
            logger.info(f"{metric_name:20s}: {value:.4f}")

        # Log to MLflow
        if mlflow_enabled:
            try:
                mlflow.log_metrics(test_metrics)
                mlflow.log_metric("best_val_score", training_results["best_score"])
                mlflow.log_metric("best_epoch", training_results["best_epoch"])
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "config": args.config,
            "seed": seed,
            "training": {
                "best_score": training_results["best_score"],
                "best_epoch": training_results["best_epoch"],
            },
            "test_metrics": test_metrics,
        }

        results_file = output_dir / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Save feature importance
        if config.get("output", {}).get("save_feature_importance", True):
            feature_importance = model.get_feature_importance_df()
            importance_file = output_dir / "feature_importance.csv"
            feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Feature importance saved to {importance_file}")

        # Generate plots
        if config.get("output", {}).get("generate_plots", True):
            try:
                from adaptive_feature_importance_reweighting_credit_risk.evaluation.analysis import (
                    generate_evaluation_report,
                )

                generate_evaluation_report(
                    y_test.values,
                    y_test_pred_proba,
                    y_test_pred,
                    feature_importance,
                    output_dir=str(output_dir),
                )
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        if mlflow_enabled:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
